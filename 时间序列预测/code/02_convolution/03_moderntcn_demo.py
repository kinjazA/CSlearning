"""
ModernTCN 完整学习示例
======================
两套内容，按需使用：

  A. PyTorch 从零构建 — 深入理解 DWConv + 倒置瓶颈 + 大kernel
  B. 合成数据训练 + 预测 + 可视化 + kernel消融实验

参考文献：
  Luo et al. "ModernTCN: A Modern Pure Convolution Structure for
  General Time Series Analysis" (ICLR 2024 Spotlight)

环境依赖：
  pip install torch numpy pandas matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# 0. 合成时序数据
# ============================================================================

def make_ts_data(num_samples=1000, L=96, H=96, C=4, seed=42):
    """生成含局部模式的多变量时间序列。

    模拟：局部模式和长程依赖混合
      var0: 温度 — 日周期(24) + AR(2)
      var1: 湿度 — 依赖温度 + 自噪声
      var2: 气压 — 独立 AR(1) 慢变
      var3: 风速 — 依赖湿度+气压+随机性

    返回 X: (N, L, C), Y: (N, H, C)
    """
    rng = np.random.default_rng(seed)
    total_len = L + H
    total_points = num_samples + total_len
    t = np.arange(total_points)

    v0 = np.zeros(total_points)
    v0[:2] = 20 + rng.normal(0, 1, 2)
    for i in range(2, total_points):
        v0[i] = 0.7*v0[i-1] - 0.2*v0[i-2] + 5*np.sin(2*np.pi*i/24) + rng.normal(0, 0.5)
    v0 = (v0 - v0.mean()) / v0.std()

    v1 = -0.6*v0 + 0.3*np.sin(2*np.pi*t/300) + rng.normal(0, 0.3, total_points)

    v2 = np.zeros(total_points)
    v2[0] = rng.normal(0, 1)
    for i in range(1, total_points):
        v2[i] = 0.9*v2[i-1] + rng.normal(0, 0.2)

    v3 = 0.3*v1 + 0.4*v2 + rng.normal(0, 0.4, total_points)

    data = np.stack([v0, v1, v2, v3], axis=-1)
    X, Y = [], []
    for start in range(num_samples):
        X.append(data[start:start+L])
        Y.append(data[start+L:start+total_len])
    return np.array(X), np.array(Y)


# ============================================================================
# A. PyTorch 从零构建 — ModernTCN
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class InvertedBottleneck(nn.Module):
    """倒置瓶颈 + 大kernel深度卷积

    设计逻辑：
      1. Expand: 升维给DWConv更多容量
      2. DWConv: 每个通道独立的时序卷积（大kernel）
      3. Squeeze: 降维回原维度 + 残差
    """
    def __init__(self, d_model, expand_ratio=3, kernel_size=51, dropout=0.1):
        super().__init__()
        hidden_dim = d_model * expand_ratio

        # 1. Expand: d → d×r  (1×1 Conv)
        self.expand = nn.Sequential(
            nn.Conv1d(d_model, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )

        # 2. DWConv: 大kernel深度卷积
        #    groups=hidden_dim → 每个通道独立的kernel
        self.dwconv = nn.Sequential(
            nn.Conv1d(
                hidden_dim, hidden_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=hidden_dim,
                bias=False
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )

        # 3. Squeeze: d×r → d  (1×1 Conv)
        self.squeeze = nn.Sequential(
            nn.Conv1d(hidden_dim, d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.Dropout(dropout)
        )

        # 残差缩放（稳定训练）
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """x: (B, d, L) → (B, d, L)"""
        residual = x
        x = self.expand(x)
        x = self.dwconv(x)
        x = self.squeeze(x)
        return residual + self.gamma * x


class ModernTCNBlock(nn.Module):
    """ModernTCN Block = 倒置瓶颈DWConv + FFN"""
    def __init__(self, d_model, expand_ratio=3, kernel_size=51, dropout=0.1):
        super().__init__()

        # 核心：大kernel深度卷积（替代Self-Attention）
        self.conv_block = InvertedBottleneck(
            d_model, expand_ratio, kernel_size, dropout
        )

        # 可选FFN（类似Transformer的FFN）
        self.ffn = nn.Sequential(
            nn.Conv1d(d_model, d_model * 4, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model * 4, d_model, kernel_size=1),
            nn.Dropout(dropout)
        )
        self.ffn_gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """x: (B, d, L) → (B, d, L)"""
        x = self.conv_block(x)
        x = x + self.ffn_gamma * self.ffn(x)
        return x


class ModernTCN(nn.Module):
    """ModernTCN: 纯现代卷积的时序预测模型

    参数:
        enc_in: 变量数
        seq_len: 回溯长度
        pred_len: 预测长度
        d_model: 特征维度
        expand_ratio: 倒置瓶颈扩张比
        kernel_size: DWConv 卷积核大小
        num_blocks: ModernTCN Block 堆叠数
    """
    def __init__(self, enc_in=7, seq_len=96, pred_len=96,
                 d_model=64, expand_ratio=3, kernel_size=51,
                 num_blocks=2, dropout=0.1):
        super().__init__()
        self.enc_in = enc_in
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.kernel_size = kernel_size

        # 输入嵌入 (Conv1d: C → d)
        self.embedding = nn.Sequential(
            nn.Conv1d(enc_in, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )

        # 堆叠 ModernTCN Block
        self.blocks = nn.ModuleList([
            ModernTCNBlock(d_model, expand_ratio, kernel_size, dropout)
            for _ in range(num_blocks)
        ])

        # 预测头: (B, d, L) → (B, H, C)
        self.predictor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),          # 全局池化: (B, d, 1)
            nn.Flatten(start_dim=1),           # (B, d)
            nn.Linear(d_model, pred_len * enc_in),
        )

    def forward(self, x):
        """
        x: (B, L, C) — channel-last
        返回: (B, H, C)
        """
        B, L, C = x.shape

        # 转 channel-first: (B, L, C) → (B, C, L)
        x = x.permute(0, 2, 1)

        # Embedding: (B, C, L) → (B, d, L)
        x = self.embedding(x)

        # ModernTCN Blocks
        for block in self.blocks:
            x = block(x)

        # 预测: (B, d, L) → (B, H*C) → (B, H, C)
        x = self.predictor(x)                     # (B, H*C)
        x = x.reshape(B, self.pred_len, self.enc_in)

        return x


# ============================================================================
# B. 训练与评估
# ============================================================================

def train_modern_tcn(model, train_loader, val_loader, epochs=50, lr=1e-3,
                     patience=10, device='cpu'):
    """训练 ModernTCN"""
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
            torch.save(model.state_dict(), 'best_moderntcn.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}')

    return train_losses, val_losses


# ============================================================================
# C. 主函数
# ============================================================================

def main():
    print("=" * 65)
    print("ModernTCN 完整演示 — 大kernel DWConv + 倒置瓶颈")
    print("=" * 65)

    torch.manual_seed(42)
    np.random.seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")

    # 1. 生成数据
    print("\n[1] 生成时序数据...")
    X, Y = make_ts_data(num_samples=1000, L=96, H=96, C=4, seed=42)
    print(f"  X: {X.shape}, Y: {Y.shape}")

    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    Y_train, Y_val = Y[:train_size], Y[train_size:]

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), torch.FloatTensor(Y_val))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 2. 构建模型
    print("\n[2] 构建 ModernTCN (kernel=51, d=64)...")
    model = ModernTCN(
        enc_in=4, seq_len=96, pred_len=96,
        d_model=64, expand_ratio=3, kernel_size=51,
        num_blocks=2, dropout=0.1
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {total_params:,}")
    print(f"  感受野: {model.kernel_size} × {2}层 ≈ {model.kernel_size*2-1} 步")

    # 3. 训练
    print("\n[3] 训练中...")
    train_losses, val_losses = train_modern_tcn(
        model, train_loader, val_loader,
        epochs=50, lr=1e-3, patience=10, device=device
    )

    # 4. 消融实验：不同 kernel 大小
    print("\n[4] 消融实验: 不同 kernel 大小...")
    k_results = {}
    for k in [7, 21, 51, 81]:
        print(f"  kernel={k}...")
        m = ModernTCN(
            enc_in=4, seq_len=96, pred_len=96,
            d_model=32, expand_ratio=2, kernel_size=k, num_blocks=1
        ).to(device)

        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        crit = nn.MSELoss()

        for _ in range(20):  # 快速训练
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
        k_results[k] = val_loss
        print(f"    kernel={k:3d}: Val MSE={val_loss:.4f}")

    # 5. 可视化
    print("\n[5] 生成可视化...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # 5a. 训练曲线
    ax = axes[0]
    ax.plot(train_losses, label='Train', linewidth=2)
    ax.plot(val_losses, label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('ModernTCN Training Curve', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 5b. Kernel 消融
    ax = axes[1]
    ks = list(k_results.keys())
    losses = [k_results[k] for k in ks]
    colors = ['#e74c3c' if k == 51 else '#3498db' for k in ks]
    ax.bar([str(k) for k in ks], losses, color=colors, edgecolor='white')
    ax.set_xlabel('Kernel Size')
    ax.set_ylabel('Val MSE')
    ax.set_title('Ablation: Kernel Size Effect', fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # 5c. 预测结果
    ax = axes[2]
    model.eval()
    with torch.no_grad():
        x_in = torch.FloatTensor(X_val[:1]).to(device)
        pred = model(x_in).squeeze(0).cpu().numpy()

    var_idx = 0
    L = X_val.shape[1]
    ax.plot(range(L), X_val[0, :, var_idx], 'gray', linewidth=1.5, label='History')
    ax.plot(range(L, L + Y_val.shape[1]), Y_val[0, :, var_idx],
            'blue', linewidth=1.5, label='True')
    ax.plot(range(L, L + pred.shape[0]), pred[:, var_idx],
            'red', linewidth=1.5, linestyle='--', label='Prediction')
    ax.axvline(x=L, color='black', linestyle=':', alpha=0.5)
    ax.set_title('Temperature Prediction', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig('moderntcn_results.png', dpi=150)
    print("  保存: moderntcn_results.png")

    # 6. 参数量对比
    print(f"\n{'='*65}")
    print(f"实验结果总结:")
    print(f"  Final Val Loss: {val_losses[-1]:.4f}")
    print(f"  参数量: {total_params:,} (极轻量!)")
    print(f"  感受野: {model.kernel_size*2-1} 步")

    best_k = min(k_results.items(), key=lambda x: x[1])
    print(f"  最优 kernel: {best_k[0]} (MSE={best_k[1]:.4f})")

    # 计算DWConv vs 标准Conv的参数量对比
    d = 64
    params_standard = 51 * d * d
    params_dwconv = 51 * d + d * d
    print(f"\n  参数量对比 (k=51, d=64):")
    print(f"    标准Conv: {params_standard:,}")
    print(f"    DWConv:   {params_dwconv:,} (仅 {params_dwconv/params_standard*100:.1f}%)")

    print(f"\n生成图片: moderntcn_results.png")
    print(f"{'='*65}")

    plt.show()


if __name__ == "__main__":
    main()
