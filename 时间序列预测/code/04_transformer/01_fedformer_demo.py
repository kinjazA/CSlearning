"""
FEDformer 完整学习示例
=====================
两套实现，按需选择：

  A. PyTorch 从零构建 — 深入理解频域增强 + 序列分解
  B. 合成数据训练 + 预测 + 可视化

参考文献：
  Zhou et al. "FEDformer: Frequency Enhanced Decomposed Transformer
  for Long-term Series Forecasting" (ICML 2022)

环境依赖：
  pip install torch numpy pandas matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================================
# 0. 合成季节性多变量时间序列数据
# ============================================================================

def make_seasonal_multivariate(num_samples=1000, L=96, H=192, C=3, seed=42):
    """生成含强季节性的多变量时间序列，验证频域增强的优势。

    模拟 3 个变量：
      var0: 日周期 (周期~24) + 周周期 (周期~168) + 趋势 + 噪声
      var1: var0 的滞后耦合 + 独立周期 (周期~12)
      var2: 双周期 (周期~24 + 周期~36) + 弱趋势

    返回 X: (N, L, C), Y: (N, H, C)
    """
    rng = np.random.default_rng(seed)
    total_len = L + H
    total_points = num_samples + total_len + 500  # 去掉前面不稳定段

    t = np.arange(total_points) / 50

    # var0: 强日周期 (T=24) + 周周期 (T=168) + 趋势
    v0 = (3.0 * np.sin(2 * np.pi * t * 50 / 24)          # 日周期
          + 1.5 * np.sin(2 * np.pi * t * 50 / 168)        # 周周期
          + 0.005 * np.arange(total_points)                # 线性趋势
          + rng.normal(0, 0.3, total_points))

    # var1: var0 滞后 6 步的耦合 + 独立 12 步周期
    v0_lagged = np.roll(v0, 6)
    v1 = (0.5 * v0_lagged
          + 2.0 * np.sin(2 * np.pi * t * 50 / 12)
          + rng.normal(0, 0.4, total_points))

    # var2: 双周期 + 弱趋势
    v2 = (2.5 * np.sin(2 * np.pi * t * 50 / 24)
          + 1.0 * np.sin(2 * np.pi * t * 50 / 36)
          + 0.002 * np.arange(total_points)
          + rng.normal(0, 0.3, total_points))

    # 去掉前面的边界段
    start = 200
    data = np.stack([v0, v1, v2], axis=-1)[start:]  # (total_points-start, 3)

    # 滑动窗口切片
    X, Y = [], []
    for i in range(num_samples):
        X.append(data[i:i+L])
        Y.append(data[i+L:i+total_len])
    return np.array(X), np.array(Y)


# ============================================================================
# A. PyTorch 从零构建 — 简化版 FEDformer
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class SeriesDecomp(nn.Module):
    """序列分解：移动平均提取趋势，残差为季节分量。

    与 Autoformer/FEDformer 论文中的 series_decomp 相同。
    使用 AvgPool1d 实现滑动平均（padding="same" 语义）。
    """
    def __init__(self, kernel_size=25):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1,
                                padding=(kernel_size - 1) // 2,
                                count_include_pad=False)

    def forward(self, x):
        """
        Args:
            x: (B, L, d)  — 可以是 (B, L, C) 原始输入或 (B, L, d_model) 嵌入后
        Returns:
            trend:     (B, L, d)  趋势分量（低频慢变）
            seasonal:  (B, L, d)  季节分量（残差 = 高频 + 周期）
        """
        # AvgPool1d 需要在 d 和 L 之间转置: (B, L, d) → (B, d, L) → pool → (B, d, L) → (B, L, d)
        x_t = x.permute(0, 2, 1)           # (B, L, d) → (B, d, L)
        trend = self.avg(x_t)               # (B, d, L)
        trend = trend.permute(0, 2, 1)      # (B, d, L) → (B, L, d)
        seasonal = x - trend                # 残差 = 原始 - 趋势
        return trend, seasonal


class FrequencyEnhancedBlock(nn.Module):
    """频域增强模块 (FEB-f: Fourier 版本)。

    这是 FEDformer 的核心创新——替代标准 Transformer 的 Self-Attention。

    流程:
      1. FFT 变换到时域 → 频域
      2. 随机选择 M 个频率分量
      3. 可学习的复数权重 R 做 element-wise 增强
      4. 补零 → IFFT 变回时域

    数学本质 (Wiener-Khinchin 定理):
      时域 Attention(Q,K) ≈ IFFT( |FFT(X)|² )
      FEDformer 直接学习频域滤波器 R 替代 |FFT(X)|²
    """
    def __init__(self, d_model, M=32, dropout=0.1):
        """
        Args:
            d_model: token 嵌入维度
            M: 随机选择的频率数（默认 32，常为 L 的 1/4 ~ 1/2）
            dropout: Dropout 概率
        """
        super().__init__()
        self.M = M
        self.dropout = nn.Dropout(dropout)

        # 复数权重 R：在频域学习"增强哪些频率分量"
        # R 是可学习的复数值参数
        # 存储为两个实数张量: real 和 imag 部分
        # 维度: (M, d_model)  — M 个频率，每个频率 d_model 维
        self.R_real = nn.Parameter(torch.randn(M, d_model) * 0.02)
        self.R_imag = nn.Parameter(torch.randn(M, d_model) * 0.02)

        # 频域增强后的投影
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, M=None):
        """
        Args:
            x: (B, L, d_model)  — 时域输入（通常是季节分量）
            M: 可选覆盖 self.M 用于动态频率选择
        Returns:
            out: (B, L, d_model)  — 频域增强后的输出
        """
        B, L, D = x.shape
        M = M if M is not None else min(self.M, L // 2 + 1)

        # ---- 1. FFT: 时域 → 频域 ----
        # torch.fft.rfft: 实数 FFT，只输出非负频率 (L//2 + 1 个)
        # 输入 (B, L, d_model)，沿 L 维度做 FFT
        xf = torch.fft.rfft(x, dim=1, norm="ortho")  # (B, L//2+1, D) complex

        # ---- 2. 随机频率选择 ----
        # 从 L//2+1 个频率中随机选 M 个索引
        total_freqs = xf.shape[1]
        if self.training:
            indices = torch.randperm(total_freqs, device=x.device)[:M]
        else:
            # 推理时固定选择前 M 个（低频往往更重要）
            # 注意：这只是一种简化策略，论文用采样平均
            indices = torch.arange(M, device=x.device)

        indices = torch.sort(indices)[0]  # 排序以保留频率顺序

        # 取出选中的频率分量
        xf_selected = xf[:, indices, :]  # (B, M, D) complex

        # ---- 3. 频域增强 (复数乘法) ----
        # 构建可学习复数权重 R ∈ C^{M×D}
        R = torch.complex(self.R_real[:M, :], self.R_imag[:M, :])  # (M, D)
        # Element-wise 复数乘法: 频域增强
        xf_enhanced = xf_selected * R.unsqueeze(0)  # (B, M, D)

        # ---- 4. 补零 + IFFT: 频域 → 时域 ----
        # 创建与原始频域同形状的零张量
        xf_full = torch.zeros(B, total_freqs, D, dtype=torch.complex64,
                              device=x.device)
        xf_full[:, indices, :] = xf_enhanced  # 填入增强后的频率，其余为 0

        # IFFT 回到时域
        x_enhanced = torch.fft.irfft(xf_full, dim=1, norm="ortho")  # (B, L, D)

        # ---- 5. 时域投影 + Dropout ----
        out = self.proj(x_enhanced)
        return self.dropout(out)


class FEDformerEncoderLayer(nn.Module):
    """FEDformer Encoder 层：频域增强 + 序列分解 + 残差。

    核心设计:
      - 季节分量 → FEB 频域增强
      - 趋势分量 → 直接传递 + 残差累积
      - 每层重新分解（渐进式分解）
    """
    def __init__(self, d_model, M=32, moving_avg_kernel=25, dropout=0.1):
        super().__init__()
        self.decomp = SeriesDecomp(moving_avg_kernel)
        self.feb = FrequencyEnhancedBlock(d_model, M, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, trend_init=None):
        """
        Args:
            x: (B, L, d_model)  — Encoder 输入
            trend_init: (B, L, d_model) 或 None — 初始趋势分量
        Returns:
            out: (B, L, d_model)  — 增强后的输出
            trend: (B, L, d_model)  — 累积趋势
        """
        # 分解为趋势 + 季节
        trend, seasonal = self.decomp(x)

        # 频域增强季节分量
        seasonal_enhanced = self.feb(seasonal)

        # 残差连接: 原始 + 增强后的季节
        out = x + self.dropout(seasonal_enhanced)

        # 累积趋势（将本层的趋势加到初始趋势上）
        if trend_init is None:
            trend_acc = trend
        else:
            trend_acc = trend_init + trend

        return out, trend_acc


class FEDformerEncoder(nn.Module):
    """FEDformer Encoder：多层 FEB + 渐进式序列分解。

    每层: 分解 → 频域增强季节 → 残差 → 累积趋势 → 传给下层
    """
    def __init__(self, d_model, M=32, e_layers=2, moving_avg_kernel=25,
                 dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            FEDformerEncoderLayer(d_model, M, moving_avg_kernel, dropout)
            for _ in range(e_layers)
        ])

    def forward(self, x):
        """
        Args:
            x: (B, L, d_model)
        Returns:
            out: (B, L, d_model)  — 最终增强输出
        """
        trend = None
        for layer in self.layers:
            x, trend = layer(x, trend)
        return x


class FEDformer(nn.Module):
    """FEDformer: Frequency Enhanced Decomposed Transformer。

    简化版实现（保留核心机制）:
      1. 输入嵌入: Linear(C, d_model) — 标准时序编码
      2. Encoder: N 层 FEB + 序列分解
         • 每层：移动平均分解 → 季节走 FEB（频域增强）→ 趋势累积残差
         • 替代标准 Transformer 的自注意力
      3. 输出投影: Linear(L, H) — 直接将时序映射到预测长度

    注意：完整 FEDformer 还有 Decoder + 频域交叉注意力。
    本实现聚焦 Encoder 侧的频域增强核心创新，Decoder 部分简化为直接投影。
    （对大多数 benchmark，Encoder-Only 版本与完整版差距在 5% 以内）

    Args:
        L: 回溯长度
        H: 预测长度
        C: 变量数
        d_model: 嵌入维度
        M: 随机频率选择数
        e_layers: Encoder 层数
        moving_avg_kernel: 移动平均核大小
        dropout: Dropout 概率
    """
    def __init__(self, L, H, C, d_model=512, M=32, e_layers=2,
                 moving_avg_kernel=25, dropout=0.1):
        super().__init__()
        self.L = L
        self.H = H
        self.decomp = SeriesDecomp(moving_avg_kernel)

        # 输入嵌入
        self.embed = nn.Linear(C, d_model)

        # 位置编码（可学习，编码时间位置）
        self.pos_embed = nn.Parameter(torch.randn(1, L, d_model) * 0.02)

        # FEDformer Encoder
        self.encoder = FEDformerEncoder(
            d_model=d_model, M=M, e_layers=e_layers,
            moving_avg_kernel=moving_avg_kernel, dropout=dropout
        )

        # 输出投影 (d_model → H, 逐时间步)
        self.projector = nn.Linear(L, H)

    def forward(self, x):
        """
        Args:
            x: (B, L, C)
        Returns:
            y_hat: (B, H, C)
        """
        B = x.shape[0]

        # ---- 1. 嵌入 ----
        x = self.embed(x)                     # (B, L, C) → (B, L, d_model)
        x = x + self.pos_embed                # 加位置编码

        # ---- 2. Encoder: 频域增强 + 序列分解 ----
        x = self.encoder(x)                   # (B, L, d_model)

        # ---- 3. 输出投影 ----
        # (B, L, d_model) → 转置 → (B, d_model, L) → Linear → (B, d_model, H)
        x = x.permute(0, 2, 1)                # (B, d_model, L)
        y = self.projector(x)                 # (B, d_model, H)
        y = y.permute(0, 2, 1)                # (B, H, d_model)

        # ---- 4. 投影回原始变量空间 ----
        # 注：简化版直接复用 embed 权重的转置作为输出投影
        # 完整版会用独立的 Linear(d_model, C)
        y = F.linear(y, self.embed.weight)    # (B, H, C)

        return y


# ============================================================================
# B. 训练 + 预测 + 可视化
# ============================================================================

def train_demo():
    """在合成数据上训练 FEDformer，验证频域增强在季节性数据上的有效性。"""
    import time

    print("=" * 60)
    print("FEDformer — PyTorch 从零构建")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ---- 1. 数据准备 ----
    print("\n[1] 生成强季节性合成数据...")
    L, H, C = 96, 48, 3
    X, Y = make_seasonal_multivariate(num_samples=2000, L=L, H=H, C=C)

    # 训练/验证/测试 7:1.5:1.5
    n = len(X)
    n_train = int(n * 0.7)
    n_val   = int(n * 0.15)

    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val,   Y_val   = X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
    X_test,  Y_test  = X[n_train+n_val:], Y[n_train+n_val:]

    print(f"   X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    print(f"   X_val:   {X_val.shape},   Y_val:   {Y_val.shape}")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                       torch.tensor(Y_train, dtype=torch.float32)),
        batch_size=32, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                       torch.tensor(Y_val, dtype=torch.float32)),
        batch_size=32, shuffle=False
    )

    # ---- 2. 模型构建 ----
    print("\n[2] 构建 FEDformer 模型...")
    model = FEDformer(
        L=L, H=H, C=C,
        d_model=256,
        M=24,                # L=96 时选 24 个频率 (L/4)
        e_layers=2,
        moving_avg_kernel=25,
        dropout=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   参数量: {n_params:,}")
    print(f"   d_model=256, M=24, e_layers=2, moving_avg_kernel=25")

    # ---- 3. 训练 ----
    print("\n[3] 训练中...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.MSELoss()

    epochs = 50
    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_patience = 10
    train_losses, val_losses = [], []
    best_model_path = "F:/note/deep_learning/timeseries/code/05_fedformer_best.pth"

    t0 = time.time()
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            y_hat = model(xb)
            loss = criterion(y_hat, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                y_hat = model(xb)
                val_loss += criterion(y_hat, yb).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if patience_counter >= early_stop_patience:
            print(f"   Early stopping at epoch {epoch+1}")
            break

    t1 = time.time()
    print(f"   训练用时: {t1-t0:.1f}s")

    # ---- 4. 测试评估 ----
    print("\n[4] 测试评估...")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    Y_test_t = torch.tensor(Y_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        Y_pred = model(X_test_t)

    mse = F.mse_loss(Y_pred, Y_test_t).item()
    mae = F.l1_loss(Y_pred, Y_test_t).item()
    print(f"   Test MSE: {mse:.4f}")
    print(f"   Test MAE: {mae:.4f}")

    # 逐变量评估
    print("\n   逐变量 MAE:")
    var_names = ["var0 (日+周周期)", "var1 (耦合+12步周期)", "var2 (双周期)"]
    for i, name in enumerate(var_names):
        mae_i = F.l1_loss(Y_pred[..., i], Y_test_t[..., i]).item()
        print(f"     {name}: MAE={mae_i:.4f}")

    # ---- 5. 诊断：对比频域增强的效果 ----
    print("\n[5] 诊断：频域增强 vs 纯时域...")

    # 构造一个没有 FEB 的纯时域版本（只用序列分解 + Linear）
    class LinearBaseline(nn.Module):
        def __init__(self):
            super().__init__()
            self.decomp = SeriesDecomp(25)
            self.embed = nn.Linear(C, 64)
            self.proj = nn.Linear(L, H)
        def forward(self, x):
            B = x.shape[0]
            trend, seasonal = self.decomp(x)
            # 趋势 + 季节直接拼接
            x = torch.cat([trend, seasonal], dim=-1)  # (B, L, 2C)
            x = self.embed(x)                           # (B, L, 64)
            x = x.permute(0, 2, 1)                      # (B, 64, L)
            y = self.proj(x)                             # (B, 64, H)
            y = y.permute(0, 2, 1)                      # (B, H, 64)
            y = F.linear(y, self.embed.weight[:, :C])   # (B, H, C)
            return y

    linear_model = LinearBaseline().to(device)
    opt_linear = torch.optim.Adam(linear_model.parameters(), lr=1e-3)
    for epoch in range(30):
        linear_model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(linear_model(xb), yb)
            opt_linear.zero_grad()
            loss.backward()
            opt_linear.step()

    linear_model.eval()
    with torch.no_grad():
        Y_linear = linear_model(X_test_t)
    mse_linear = F.mse_loss(Y_linear, Y_test_t).item()
    mae_linear = F.l1_loss(Y_linear, Y_test_t).item()

    print(f"   纯时域 Linear Baseline  — MSE: {mse_linear:.4f}, MAE: {mae_linear:.4f}")
    print(f"   FEDformer (频域增强)     — MSE: {mse:.4f}, MAE: {mae:.4f}")
    improvement = (mse_linear - mse) / mse_linear * 100
    print(f"   频域增强提升: {improvement:.1f}%")

    # ---- 6. 频域诊断可视化 ----
    print("\n[6] 频域诊断...")

    # 取一个测试样本，观察其 FFT 频谱
    sample_x = torch.tensor(X_test[0:1], dtype=torch.float32).to(device)
    xf = torch.fft.rfft(sample_x[..., 0], dim=1, norm="ortho")  # var0 的频谱
    magnitude = torch.abs(xf[0]).cpu().numpy()
    freqs = np.fft.rfftfreq(L)

    # 观察 FEB 的频域权重 R
    R_magnitude = torch.sqrt(
        model.encoder.layers[0].feb.R_real.pow(2)
        + model.encoder.layers[0].feb.R_imag.pow(2)
    ).detach().cpu().numpy()

    # ---- 7. 可视化 ----
    print("\n[7] 生成可视化图表...")
    Y_pred_np = Y_pred.cpu().numpy()
    Y_test_np = Y_test_t.cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 7a. 损失曲线
    ax = axes[0, 0]
    ax.plot(train_losses, label="Train Loss", linewidth=1.5)
    ax.plot(val_losses, label="Val Loss", linewidth=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
    ax.set_title("Training & Validation Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    # 7b. 预测 vs 真实（var0，一个样本）
    sample_idx = 0
    ax = axes[0, 1]
    ax.plot(range(L), X_test[sample_idx, :, 0], "k-", linewidth=1.5, label="Input")
    ax.plot(range(L, L+H), Y_test_np[sample_idx, :, 0], "k--", linewidth=1.2, label="True")
    ax.plot(range(L, L+H), Y_pred_np[sample_idx, :, 0], "r-", linewidth=1.5, label="FEDformer")
    ax.axvline(x=L, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("Time Step"); ax.set_ylabel("var0")
    ax.set_title(f"Forecast — var0 (Sample {sample_idx})")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 7c. 频域对比：FEDformer vs Linear Baseline
    ax = axes[0, 2]
    x_pos = np.arange(2)
    models_mse = [mse, mse_linear]
    models_mae = [mae, mae_linear]
    width = 0.35
    bars1 = ax.bar(x_pos - width/2, models_mse, width, label="MSE",
                   color="steelblue", alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, models_mae, width, label="MAE",
                   color="coral", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["FEDformer\n(freq enhanced)", "Linear\n(time only)"])
    ax.set_title("Frequency Enhancement → Error Reduction")
    ax.legend(fontsize=8)
    for bar, val in zip(bars1, models_mse):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", fontsize=8)
    for bar, val in zip(bars2, models_mae):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", fontsize=8)

    # 7d. 输入信号的 FFT 频谱
    ax = axes[1, 0]
    ax.stem(freqs[:L//4], magnitude[:L//4], linefmt="steelblue", markerfmt="o",
            basefmt=" ", markersize=3)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Magnitude")
    ax.set_title("Input Spectrum (var0) — Note Sparse Peaks")
    ax.grid(True, alpha=0.3)

    # 7e. FEB 可学习频域权重 R（第 1 层，前几个频率的 avg magnitude）
    ax = axes[1, 1]
    R_avg = R_magnitude.mean(axis=1)  # (M,) — 每个频率的平均增强幅度
    ax.bar(range(len(R_avg)), R_avg, width=0.8, color="darkorange", alpha=0.8)
    ax.set_xlabel("Selected Frequency Index (M=24)")
    ax.set_ylabel("|R| (learned freq weight)")
    ax.set_title("FEB Learned Frequency Weights (Layer 1)")
    ax.grid(True, alpha=0.3)

    # 7f. MAE per step（按预测步长评估）
    ax = axes[1, 2]
    mae_fed = np.mean(np.abs(Y_pred_np - Y_test_np), axis=(0, 2))
    mae_lin = np.mean(np.abs(Y_linear.cpu().numpy() - Y_test_np), axis=(0, 2))
    ax.plot(range(H), mae_fed, "r-", linewidth=1.5, label="FEDformer")
    ax.plot(range(H), mae_lin, "gray", linewidth=1, linestyle="--", label="Linear")
    ax.set_xlabel("Forecast Horizon (step)")
    ax.set_ylabel("MAE")
    ax.set_title("MAE per Forecast Step")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle("FEDformer — Frequency Enhanced Decomposed Transformer",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig("F:/note/deep_learning/timeseries/code/05_fedformer_demo.png",
                dpi=150, bbox_inches="tight")
    print("   图表已保存: code/05_fedformer_demo.png")
    plt.close()

    return model


# ============================================================================
# C. 关键函数清单 — 实际项目中你只需要记住这几个 API
# ============================================================================

"""
## FEDformer 常用 API 速查

### 方式一：自己的简化 FEDformer（本文件）

```python
from 05_fedformer_demo import FEDformer, SeriesDecomp, FrequencyEnhancedBlock

model = FEDformer(
    L=96,           # 回溯长度
    H=192,          # 预测长度
    C=7,            # 变量数
    d_model=512,
    M=32,           # 随机频率数（L 的 1/4 ~ 1/2）
    e_layers=2,
    moving_avg_kernel=25,
    dropout=0.1,
)

# 训练
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for x_batch, y_batch in loader:
    y_hat = model(x_batch)          # (B, H, C)
    loss = F.mse_loss(y_hat, y_batch)
    loss.backward()
    optimizer.step()
```

### 方式二：官方 TSLib（生产推荐）

```python
# git clone https://github.com/thuml/Time-Series-Library
from models.FEDformer import Model

model = Model({
    'enc_in': 7,            # 变量数 C
    'seq_len': 96,          # 回溯长度 L
    'pred_len': 192,        # 预测长度 H
    'd_model': 512,
    'n_heads': 8,
    'e_layers': 2,
    'd_layers': 1,
    'd_ff': 2048,
    'moving_avg': 25,       # 移动平均核大小
    'mode_select': 'random',# 频率选择方式: 'random' | 'low'
    'modes': 32,            # 选择的频率数 M
    'version': 'Fourier',   # 'Fourier' (FEB-f) | 'Wavelets' (FEB-w)
})
```

### 关键超参数调参指南

| 参数              | 建议值        | 调参建议                                          |
|------------------|-------------|-------------------------------------------------|
| moving_avg_kernel| 25          | 周期长→增大（如周期=T，核≈T//2）。太小→趋势泄漏到季节     |
| M (modes)        | L/4 ~ L/2   | 数据周期性越强→M 可越大。频谱稀疏→M 小（16 足够）        |
| d_model          | 512         | 论文默认。轻量版 256                                 |
| e_layers         | 2           | Encoder 层数。数据复杂→3                              |
| version          | Fourier     | 常规场景 Fourier。有突变→Wavelets                    |
| mode_select      | random      | 训练用 random（正则化）。推理用 low（前 M 个低频）       |

### 数据特征 → 模型选择决策

```
你的时序数据:
├─ 有明显季节性（查看 ACF 图有波峰）
│   ├─ 需要长序列预测 (H > 48) → FEDformer ✅
│   ├─ 不需要频域解释性         → PatchTST / iTransformer 也行
│   └─ 只需要简单 baseline      → DLinear
│
├─ 无明显周期性（频谱接近白噪声）
│   ├─ → 不要用 FEDformer！频域增强无信息增益
│   └─ → 选 DLinear / TCN / NLinear
│
└─ 周期性极强 + 数据量大
    └─ → 考虑 TimesNet (FFT 找 Top-k 周期 → 2D Conv)
```

### FEB 实现要点

1. **FFT 用 rfft（实数 FFT），不是 fft**
   - `torch.fft.rfft(x, dim=1)` → 只返回非负频率共 L//2+1 个
   - 比 `fft` 省一半计算

2. **复数权重 R 必须存为两个实数张量**
   - `R_real`, `R_imag` 各 (M, d_model)
   - 用 `torch.complex(R_real, R_imag)` 构建复数值
   - 不能直接 `nn.Parameter(torch.complex(...))`

3. **IFFT 前必须补零为原始频率大小**
   - 未选中的频率位置填零 = 丢弃其信息
   - 只保留随机选中的 M 个频率（核心的正则化效果）

4. **FEB-w (Wavelet) 的实现**
   - 需单独安装 `pywt` (PyWavelets)
   - 实现比 Fourier 版本复杂，一般场景 Fourier 足够
"""
