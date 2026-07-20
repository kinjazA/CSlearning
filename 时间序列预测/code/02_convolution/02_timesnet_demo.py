"""
TimesNet 完整学习示例
=====================
两套内容，按需使用：

  A. PyTorch 从零构建 — 深入理解 FFT→1D→2D→Conv→2D→1D 流程
  B. 合成数据训练 + 预测 + 可视化 + 频谱分析

参考文献：
  Wu et al. "TimesNet: Temporal 2D-Variation Modeling for General
  Time Series Analysis" (ICLR 2023)

环境依赖：
  pip install torch numpy pandas matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# 0. 合成多周期时间序列数据
# ============================================================================

def make_multi_period_ts(num_samples=1000, L=96, H=96, C=4, seed=42):
    """生成含多个周期的多变量时间序列。

    模拟 4 个变量，每个包含多种周期：
      var0: 温度 — 日周期(24) + 半日周期(12) + 趋势
      var1: 湿度 — 日周期(24) + 长周期(48) + 依赖温度
      var2: 气压 — 弱周期(24) + 大噪声
      var3: 风速 — 依赖湿度+气压，带随机性

    返回 X: (N, L, C), Y: (N, H, C)
    """
    rng = np.random.default_rng(seed)
    total_len = L + H
    total_points = num_samples + total_len
    t = np.arange(total_points)

    # var0: 温度 — 日周期(24) + 半日周期(12)
    daily = 10 * np.sin(2 * np.pi * t / 24)
    half_daily = 3 * np.sin(2 * np.pi * t / 12)
    trend = t * 0.01
    v0 = daily + half_daily + trend + rng.normal(0, 0.5, total_points)

    # var1: 湿度 — 日周期 + 长周期(48) + 受温度影响
    daily_h = 5 * np.sin(2 * np.pi * t / 24 + 0.5)
    long_h = 3 * np.sin(2 * np.pi * t / 48)
    v1 = daily_h + long_h - 0.3 * v0 + rng.normal(0, 0.8, total_points)

    # var2: 气压 — 弱日周期 + 大噪声
    v2 = 2 * np.sin(2 * np.pi * t / 24 + 1.0) + rng.normal(0, 1.5, total_points)

    # var3: 风速 — 依赖湿度+气压
    v3 = 0.4 * v1 + 0.3 * v2 + rng.normal(0, 0.6, total_points)

    # 标准化
    for v in [v0, v1, v2, v3]:
        v = (v - v.mean()) / v.std()

    # 滑动窗口
    data = np.stack([v0, v1, v2, v3], axis=-1)
    X, Y = [], []
    for start in range(num_samples):
        X.append(data[start:start+L])
        Y.append(data[start+L:start+total_len])
    return np.array(X), np.array(Y)


# ============================================================================
# A. PyTorch 从零构建 — TimesNet
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class InceptionBlock2D(nn.Module):
    """多尺度 2D Inception 卷积模块

    用 1×1, 3×3, 5×5 三种 kernel 同时捕捉不同尺度的 2D 模式。
    借鉴 GoogLeNet 的 Inception 设计。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        oc = out_channels // 3

        self.conv1 = nn.Conv2d(in_channels, oc, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, oc, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, oc, kernel_size=5, padding=2)

        # 调整 channel 数（整除不完时）
        remainder = out_channels - 3 * oc
        self.conv_rem = nn.Conv2d(in_channels, remainder, kernel_size=1) if remainder > 0 else None

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        """x: (B, C, H, W)"""
        c1 = self.conv1(x)
        c3 = self.conv3(x)
        c5 = self.conv5(x)
        out = torch.cat([c1, c3, c5], dim=1)
        if self.conv_rem is not None:
            out = torch.cat([out, self.conv_rem(x)], dim=1)
        return self.act(self.bn(out))


class TimesBlock(nn.Module):
    """TimesNet 核心模块: FFT找周期 → 1D→2D → 2D Conv → 2D→1D → 多周期融合"""
    def __init__(self, seq_len, d_model, top_k=5, d_ff=32):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.top_k = top_k

        # 2D Inception 卷积（所有周期共享参数）
        self.conv = nn.Sequential(
            InceptionBlock2D(d_model, d_ff),
            InceptionBlock2D(d_ff, d_model),
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (B, L, d)
        返回: (B, L, d)
        """
        B, L, d = x.shape
        residual = x

        # 1. FFT 找 Top-K 周期
        x_fft = torch.fft.rfft(x, dim=1)
        amplitude = torch.abs(x_fft).mean(dim=(0, -1))  # 平均所有通道

        # 排除频率0（直流分量），选 Top-K
        k = min(self.top_k, len(amplitude) - 1)
        _, topk_indices = torch.topk(amplitude[1:], k=k)
        topk_indices = topk_indices + 1  # 补偿偏移

        periods = L // topk_indices
        amplitudes_selected = amplitude[topk_indices]

        # 2. 过滤非法周期
        valid_mask = (periods >= 2) & (periods <= L // 2)
        periods = periods[valid_mask]
        amplitudes_selected = amplitudes_selected[valid_mask]

        if len(periods) == 0:
            return self.norm(residual)

        # 3. 对每个周期做 1D→2D→Conv→2D→1D
        outputs = []
        for i, p in enumerate(periods):
            p = p.item()

            # 填充到周期整数倍
            length = ((L - 1) // p + 1) * p
            pad_len = length - L
            x_padded = F.pad(x, (0, 0, 0, pad_len))

            # 1D → 2D: (B, length, d) → (B, d, length//p, p)
            n_rows = length // p
            x_2d = x_padded.reshape(B, n_rows, p, d)
            x_2d = x_2d.permute(0, 3, 1, 2)  # (B, d, rows, cols)

            # 2D Inception Conv
            x_2d = self.conv(x_2d)

            # 2D → 1D: (B, d, rows, cols) → (B, L, d)
            x_1d = x_2d.permute(0, 2, 3, 1)  # (B, rows, cols, d)
            x_1d = x_1d.reshape(B, length, d)
            x_1d = x_1d[:, :L, :]  # 截断回原长度

            outputs.append(x_1d)

        # 4. 加权融合：振幅越大的周期贡献越大
        weights = F.softmax(amplitudes_selected[:len(outputs)], dim=0)
        out = sum(w * o for w, o in zip(weights, outputs))

        return self.norm(residual + out)


class TimesNet(nn.Module):
    """TimesNet: 1D 时序 → 2D 图像的多周期建模

    参数:
        enc_in: 变量数
        seq_len: 回溯长度
        pred_len: 预测长度
        d_model: 特征维度（通常 32-64，比 Transformer 小）
        d_ff: Inception 内部维度
        top_k: FFT 选多少个周期
        num_blocks: TimesBlock 堆叠数
    """
    def __init__(self, enc_in=7, seq_len=96, pred_len=96,
                 d_model=32, d_ff=32, top_k=5, num_blocks=2):
        super().__init__()
        self.enc_in = enc_in
        self.pred_len = pred_len

        # Embedding
        self.embedding = nn.Linear(enc_in, d_model)

        # 堆叠 TimesBlock
        self.blocks = nn.ModuleList([
            TimesBlock(seq_len, d_model, top_k, d_ff)
            for _ in range(num_blocks)
        ])

        # 预测头
        self.predictor = nn.Linear(seq_len * d_model, pred_len * enc_in)

    def forward(self, x):
        """x: (B, L, C) → (B, H, C)"""
        B, L, C = x.shape

        # Embedding: (B, L, C) → (B, L, d)
        x = self.embedding(x)

        # TimesBlocks
        for block in self.blocks:
            x = block(x)

        # Flatten + 预测
        x = x.reshape(B, -1)
        x = self.predictor(x)
        x = x.reshape(B, self.pred_len, self.enc_in)

        return x

    def get_periods(self, x):
        """调试用：查看 FFT 发现了哪些周期"""
        self.eval()
        with torch.no_grad():
            x_fft = torch.fft.rfft(self.embedding(x), dim=1)
            amplitude = torch.abs(x_fft).mean(dim=(0, -1))
            _, indices = torch.topk(amplitude[1:], k=5)
            periods = x.shape[1] // (indices + 1)
        return periods.tolist()


# ============================================================================
# B. 训练与评估
# ============================================================================

def train_timesnet(model, train_loader, val_loader, epochs=50, lr=1e-3,
                   patience=10, device='cpu'):
    """训练 TimesNet"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_timesnet.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}')

    return train_losses, val_losses


def visualize_spectrum(model, X, device='cpu'):
    """可视化 FFT 频谱——看看 TimesNet '看到'了什么周期"""
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(X[:1]).to(device)
        x_emb = model.embedding(x)
        x_fft = torch.fft.rfft(x_emb, dim=1)
        amplitude = torch.abs(x_fft).mean(dim=(0, -1)).cpu().numpy()

    L = X.shape[1]
    freqs = np.arange(len(amplitude))
    periods = L / (freqs + 1e-8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    # 左：频率-振幅
    ax1.stem(freqs[:50], amplitude[:50])
    ax1.set_xlabel('Frequency Index')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('FFT Spectrum (前50个频率)', fontweight='bold')
    ax1.grid(alpha=0.3)

    # 标注 Top-5 峰值
    top5_idx = np.argsort(amplitude[1:])[-5:] + 1
    for idx in top5_idx:
        ax1.annotate(f'P={L//idx:.0f}', (idx, amplitude[idx]),
                     textcoords="offset points", xytext=(0, 10),
                     fontsize=9, color='red', ha='center')

    # 右：周期-振幅
    valid = (periods >= 2) & (periods <= L // 2) & (periods == periods.astype(int))
    ax2.stem(periods[valid][:30], amplitude[valid][:30])
    ax2.set_xlabel('Period Length')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Period vs Amplitude', fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.invert_xaxis()  # 短周期在左

    plt.tight_layout()
    return fig


# ============================================================================
# C. 主函数
# ============================================================================

def main():
    print("=" * 65)
    print("TimesNet 完整演示")
    print("=" * 65)

    torch.manual_seed(42)
    np.random.seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")

    # 1. 生成多周期数据
    print("\n[1] 生成多周期时间序列 (日周期=24, 半日=12, 长周期=48)...")
    X, Y = make_multi_period_ts(num_samples=1000, L=96, H=96, C=4, seed=42)
    print(f"  X: {X.shape}, Y: {Y.shape}")

    # 划分数据
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    Y_train, Y_val = Y[:train_size], Y[train_size:]

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), torch.FloatTensor(Y_val))

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 2. 频谱可视化
    print("\n[2] 可视化 FFT 频谱...")
    model_tmp = TimesNet(enc_in=4, seq_len=96, pred_len=96).to(device)
    fig_spec = visualize_spectrum(model_tmp, X_train, device)
    fig_spec.savefig('timesnet_spectrum.png', dpi=150)
    print("  保存: timesnet_spectrum.png")

    # 3. 查看发现的周期
    periods = model_tmp.get_periods(torch.FloatTensor(X_train[:1]).to(device))
    print(f"  FFT 发现的 Top-5 周期: {periods}")
    print(f"  期望的周期: 24(日), 12(半日), 48(长周期)")

    # 4. 构建模型并训练
    print("\n[3] 构建 TimesNet 并训练...")
    model = TimesNet(
        enc_in=4, seq_len=96, pred_len=96,
        d_model=32, d_ff=32, top_k=5, num_blocks=2
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {total_params:,}")

    train_losses, val_losses = train_timesnet(
        model, train_loader, val_loader,
        epochs=50, lr=1e-3, patience=10, device=device
    )

    # 5. 消融实验：不同 top-K
    print("\n[4] 消融实验: 不同 top-K...")
    k_results = {}
    for K in [1, 3, 5, 7, 10]:
        m = TimesNet(enc_in=4, seq_len=96, pred_len=96,
                     d_model=16, d_ff=16, top_k=K, num_blocks=1).to(device)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        crit = nn.MSELoss()

        for _ in range(15):  # 快速训练
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                loss = crit(m(x), y)
                opt.zero_grad()
                loss.backward()
                opt.step()

        m.eval()
        val_loss = sum(
            crit(m(x.to(device)), y.to(device)).item()
            for x, y in val_loader
        ) / len(val_loader)
        k_results[K] = val_loss
        print(f"  K={K:2d}: Val MSE={val_loss:.4f}")

    # 6. 可视化
    print("\n[5] 生成可视化...")

    fig2, axes = plt.subplots(1, 3, figsize=(16, 4))

    # 6a. 训练曲线
    ax = axes[0]
    ax.plot(train_losses, label='Train', linewidth=2)
    ax.plot(val_losses, label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('TimesNet Training Curve', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 6b. 消融实验
    ax = axes[1]
    ks = list(k_results.keys())
    losses = [k_results[k] for k in ks]
    ax.bar(ks, losses, color=['#e74c3c' if k == 5 else '#3498db' for k in ks])
    ax.set_xlabel('Top-K')
    ax.set_ylabel('Val MSE')
    ax.set_title('Ablation: Top-K Effect', fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # 6c. 预测结果
    ax = axes[2]
    model.eval()
    with torch.no_grad():
        x_in = torch.FloatTensor(X_val[:1]).to(device)
        pred = model(x_in).squeeze(0).cpu().numpy()

    var_idx = 0
    L = X_val.shape[1]
    ax.plot(range(L), X_val[0, :, var_idx], label='History', color='gray', linewidth=1.5)
    ax.plot(range(L, L + Y_val.shape[1]), Y_val[0, :, var_idx],
            label='True', color='blue', linewidth=1.5)
    ax.plot(range(L, L + pred.shape[0]), pred[:, var_idx],
            label='Prediction', color='red', linewidth=1.5, linestyle='--')
    ax.axvline(x=L, color='black', linestyle=':', alpha=0.5)
    ax.set_title('Temperature Prediction', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig2.savefig('timesnet_results.png', dpi=150)
    print("  保存: timesnet_results.png")

    # 总结
    best_k = min(k_results.items(), key=lambda x: x[1])
    discovered = model_tmp.get_periods(torch.FloatTensor(X_train[:1]).to(device))
    print(f"\n{'='*65}")
    print(f"实验结果总结:")
    print(f"  FFT 发现周期: {discovered}")
    print(f"  最优 Top-K: {best_k[0]} (MSE={best_k[1]:.4f})")
    print(f"  最终 Val Loss: {val_losses[-1]:.4f}")
    print(f"  参数量: {total_params:,} (非常轻量!)")
    print(f"\n生成图片:")
    print(f"  - timesnet_spectrum.png: FFT频谱 + 周期分析")
    print(f"  - timesnet_results.png: 训练曲线 + 消融 + 预测")
    print(f"{'='*65}")

    plt.show()


if __name__ == "__main__":
    main()
