"""
iTransformer 完整学习示例
==========================
两套实现，按需选择：

  A. PyTorch 从零构建 — 深入理解"倒置 Transformer"的内部机制
  B. 合成数据训练 + 预测 + 可视化

参考文献：
  Liu et al. "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting" (ICLR 2024)

环境依赖：
  pip install torch numpy pandas matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================================
# 0. 合成多变量时间序列数据
# ============================================================================

def make_multivariate_ts(num_samples=1000, L=96, H=96, C=4, seed=42):
    """生成含跨变量依赖的多变量时间序列。

    模拟 4 个物理量：
      var0: 温度（自回归 + 季节）
      var1: 湿度（依赖 var0 + 自噪声）
      var2: 气压（独立）
      var3: 风速（依赖 var1 + var2 + 噪声）

    返回 X: (N, L, C), Y: (N, H, C)
    """
    rng = np.random.default_rng(seed)
    total_len = L + H
    total_points = num_samples + total_len

    # ---- 生成基础序列 ----
    t = np.arange(total_points) / 50
    # var0: 温度（AR(2) + 季节）
    v0 = np.zeros(total_points)
    v0[:2] = 20 + rng.normal(0, 1, 2)
    for i in range(2, total_points):
        v0[i] = 0.7 * v0[i-1] - 0.2 * v0[i-2] + 5 * np.sin(2*np.pi*i/200) + rng.normal(0, 0.5)
    v0 = (v0 - v0.mean()) / v0.std()

    # var1: 湿度（受温度反向影响：温度高 → 湿度低）
    v1 = -0.6 * v0 + 0.3 * np.sin(2*np.pi*t/300) + rng.normal(0, 0.3, total_points)

    # var2: 气压（独立 AR(1)）
    v2 = np.zeros(total_points)
    v2[0] = rng.normal(0, 1)
    for i in range(1, total_points):
        v2[i] = 0.9 * v2[i-1] + rng.normal(0, 0.2)

    # var3: 风速（受湿度和气压影响）
    v3 = 0.3 * v1 + 0.4 * v2 + rng.normal(0, 0.4, total_points)

    # ---- 滑动窗口切片 ----
    data = np.stack([v0, v1, v2, v3], axis=-1)  # (total_points, 4)
    X, Y = [], []
    for start in range(num_samples):
        x = data[start:start+L]
        y = data[start+L:start+total_len]
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)


# ============================================================================
# A. PyTorch 从零构建 — 完整 iTransformer
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class RevIN(nn.Module):
    """可逆实例归一化 (Reversible Instance Normalization)

    论文: Kim et al. "Reversible Instance Normalization for Accurate
          Time-Series Forecasting against Distribution Shift" (ICLR 2022)

    原理：训练前对每个变量独立做 z-score 归一化，消除非平稳性；
         预测后把归一化"撤销"，恢复原始量纲。

    iTransformer 论文使用 RevIN 作为标配预处理。
    """
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.affine = True
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias   = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode="norm"):
        """
        Args:
            x: (B, L, C) 或 (B, H, C)
            mode: "norm" 归一化 或 "denorm" 反归一化
        Returns:
            normalized x, same shape
        """
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True)      # (B, 1, C)
            self.std  = x.std(dim=1, keepdim=True) + self.eps
            x = (x - self.mean) / self.std
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        elif mode == "denorm":
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self.std + self.mean
            return x
        else:
            raise ValueError(f"Unknown mode: {mode}")


class DataEmbeddingInverted(nn.Module):
    """倒置嵌入层：每个变量的 L 步历史 → d_model 维 token。

    标准 Transformer:   Linear(C, d_model) — 每个时间步投影
    iTransformer:       Linear(L, d_model) — 每个变量投影（倒置！）

    Shape flow:
        输入: (B, C, L)  ← 已经转置，每个变量一行
        输出: (B, C, d_model)  ← C 个 token，每个 d_model 维
    """
    def __init__(self, L, d_model, dropout=0.1):
        """
        Args:
            L: 回溯长度（每个变量 token 的原始维度）
            d_model: token 表示维度
        """
        super().__init__()
        self.value_embedding = nn.Linear(L, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, L)
        out = self.value_embedding(x)   # (B, C, d_model)
        return self.dropout(out)


class iTransformerBlock(nn.Module):
    """iTransformer 的一个 Encoder Block。

    两个子层，均沿变量维度操作：
      (1) Multi-Head Self-Attention — C 个 token 之间互相注意
          → 学习"温度对湿度有什么影响？"
      (2) Feed-Forward Network — 每个 token 独立通过
          → 学习"这个变量的过去模式怎么映射到未来？"

    注意 LayerNorm 的维度：沿 d_model 维度归一化每个 token，
    这与标准 Transformer 一致（只是 token 的意义变了）。
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, C, d_model) — C 个 token，每个 d_model 维
        Returns:
            out: (B, C, d_model)
        """
        # ---- 子层 1: 变量维度的 Multi-Head Self-Attention ----
        # Q, K, V 都来自 x，计算 C 个 token 间的注意力
        attn_out, _ = self.attention(x, x, x)          # (B, C, d_model)
        x = self.norm1(x + self.dropout(attn_out))      # (B, C, d_model)

        # ---- 子层 2: FFN（逐 token，共享权重） ----
        ffn_out = self.ffn(x)                            # (B, C, d_model)
        x = self.norm2(x + self.dropout(ffn_out))        # (B, C, d_model)

        return x


class iTransformer(nn.Module):
    """iTransformer: Inverted Transformer for Time Series Forecasting.

    核心架构（倒置 Transformer）:
      Input:  (B, L, C) — 回溯 L 步，C 个变量
      Embed:  转置 → (B, C, L) → Linear(L, d_model) → (B, C, d_model)
      Encoder: N 层 iTransformerBlock
      Project: Linear(d_model, H) → (B, C, H) → 转置 → (B, H, C)

    与标准 Transformer 的关键差异:
      - token = 变量（不是时间步）
      - Attention = 跨变量（不是跨时间）
      - FFN = 逐变量学时间模式（不是逐时间步做特征变换）
    """
    def __init__(self, L, H, C, d_model=512, n_heads=8, e_layers=3,
                 d_ff=2048, dropout=0.1, use_revin=True):
        """
        Args:
            L: 回溯窗口长度
            H: 预测长度
            C: 变量数
            d_model: token 嵌入维度
            n_heads: 多头注意力头数
            e_layers: Encoder Block 层数
            d_ff: FFN 隐层维度
            dropout: Dropout 概率
            use_revin: 是否使用 RevIN 归一化
        """
        super().__init__()
        self.L = L
        self.H = H
        self.C = C

        # RevIN — 实例归一化（消除非平稳性）
        self.revin = RevIN(C) if use_revin else None

        # 倒置嵌入：每个变量的 L 步历史 → d_model 维 token
        self.embed = DataEmbeddingInverted(L, d_model, dropout)

        # N 层 iTransformer Encoder Block
        self.blocks = nn.ModuleList([
            iTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(e_layers)
        ])

        # 输出投影：d_model → H（每个变量的 token → 其未来 H 步）
        self.projector = nn.Linear(d_model, H)

    def forward(self, x):
        """
        Args:
            x: (B, L, C) — 回溯窗口
        Returns:
            y_hat: (B, H, C) — 预测
        """
        B = x.shape[0]

        # ---- 1. RevIN 归一化 ----
        if self.revin is not None:
            x = self.revin(x, mode="norm")   # (B, L, C)

        # ---- 2. 倒置嵌入 ----
        # 转置：时间步维度 ↔ 变量维度
        # 标准 Transformer: (B, L, C) → 每个时间步 embed
        # iTransformer:      (B, C, L) → 每个变量 embed   ← 倒置！
        x = x.permute(0, 2, 1)                # (B, L, C) → (B, C, L)
        x = self.embed(x)                      # (B, C, L) → (B, C, d_model)

        # ---- 3. 变量维度 Self-Attention ----
        # Attention 在 C 个 token 之间计算（不是 L 个！）
        for block in self.blocks:
            x = block(x)                       # (B, C, d_model)

        # ---- 4. 输出投影 ----
        y = self.projector(x)                  # (B, C, d_model) → (B, C, H)
        y = y.permute(0, 2, 1)                 # (B, C, H) → (B, H, C)

        # ---- 5. RevIN 反归一化 ----
        if self.revin is not None:
            y = self.revin(y, mode="denorm")

        return y


# ============================================================================
# B. 训练 + 预测 + 可视化
# ============================================================================

def train_demo():
    """在合成数据上训练 iTransformer，演示完整流程。"""
    import time

    print("=" * 60)
    print("iTransformer — PyTorch 从零构建")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ---- 1. 数据准备 ----
    print("\n[1] 生成合成多变量数据...")
    L, H, C = 96, 96, 4
    X, Y = make_multivariate_ts(num_samples=2000, L=L, H=H, C=C)

    # 训练/验证/测试 7:1.5:1.5
    n = len(X)
    n_train = int(n * 0.7)
    n_val   = int(n * 0.15)

    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val,   Y_val   = X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
    X_test,  Y_test  = X[n_train+n_val:], Y[n_train+n_val:]

    print(f"   X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    print(f"   X_val:   {X_val.shape},   Y_val:   {Y_val.shape}")
    print(f"   X_test:  {X_test.shape},  Y_test:  {Y_test.shape}")

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
    print("\n[2] 构建 iTransformer 模型...")
    model = iTransformer(
        L=L, H=H, C=C,
        d_model=256,      # 减小以便快速训练
        n_heads=8,
        e_layers=3,
        d_ff=512,
        dropout=0.1,
        use_revin=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   参数量: {n_params:,}")
    print(f"   d_model=256, n_heads=8, e_layers=3, d_ff=512")

    # ---- 3. 训练 ----
    print("\n[3] 训练中...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    epochs = 50
    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_patience = 10
    train_losses, val_losses = [], []

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

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(),
                       "F:/note/deep_learning/timeseries/code/04_itransformer_best.pth")
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
    model.load_state_dict(torch.load(
        "F:/note/deep_learning/timeseries/code/04_itransformer_best.pth",
        weights_only=True
    ))
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
    var_names = ["温度 (v0)", "湿度 (v1)", "气压 (v2)", "风速 (v3)"]
    for i, name in enumerate(var_names):
        mae_i = F.l1_loss(Y_pred[..., i], Y_test_t[..., i]).item()
        corr_i = np.corrcoef(
            Y_pred[..., i].cpu().numpy().flatten(),
            Y_test_t[..., i].cpu().numpy().flatten()
        )[0, 1]
        print(f"     {name}: MAE={mae_i:.4f}, Corr={corr_i:.3f}")

    # ---- 5. 可视化 ----
    print("\n[5] 生成可视化图表...")
    Y_pred_np = Y_pred.cpu().numpy()
    Y_test_np = Y_test_t.cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 5a. 损失曲线
    ax = axes[0, 0]
    ax.plot(train_losses, label="Train Loss", linewidth=1.5)
    ax.plot(val_losses, label="Val Loss", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5b. 预测 vs 真实（第一个样本，温度变量）
    sample_idx = 0
    var_idx = 0
    ax = axes[0, 1]
    ax.plot(range(L), X_test[sample_idx, :, var_idx], "k-", linewidth=1.5,
            label="Input (Lookback)")
    ax.plot(range(L, L+H), Y_test_np[sample_idx, :, var_idx], "k--", linewidth=1.2,
            label="Ground Truth")
    ax.plot(range(L, L+H), Y_pred_np[sample_idx, :, var_idx], "r-", linewidth=1.5,
            label="iTransformer")
    ax.axvline(x=L, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Temperature (normalized)")
    ax.set_title(f"Forecast — Temperature (Sample {sample_idx})")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 5c. 多变量预测（第一个样本，所有变量叠加）
    ax = axes[0, 2]
    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]
    for vi in range(C):
        ax.plot(range(L+H), np.concatenate(
            [X_test[sample_idx, :, vi], Y_test_np[sample_idx, :, vi]]
        ), color=colors[vi], linewidth=1, alpha=0.6)
        ax.plot(range(L, L+H), Y_pred_np[sample_idx, :, vi],
                color=colors[vi], linewidth=1.5, linestyle="--")
    ax.axvline(x=L, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("Time Step")
    ax.set_title("All Variables — One Sample")
    ax.grid(True, alpha=0.3)

    # 5d. MAE per step（按预测步长评估）
    ax = axes[1, 0]
    mae_per_step = np.mean(np.abs(Y_pred_np - Y_test_np), axis=(0, 2))
    ax.bar(range(H), mae_per_step, width=1.0, alpha=0.7, color="steelblue")
    ax.set_xlabel("Forecast Horizon (step)")
    ax.set_ylabel("MAE")
    ax.set_title("MAE per Forecast Step")
    ax.grid(True, alpha=0.3)

    # 5e. 逐变量 MAE 对比
    ax = axes[1, 1]
    mae_per_var = [F.l1_loss(Y_pred[..., i], Y_test_t[..., i]).item()
                   for i in range(C)]
    bars = ax.bar(range(C), mae_per_var, color=colors, alpha=0.8)
    ax.set_xticks(range(C))
    ax.set_xticklabels(["Temp", "Humidity", "Pressure", "Wind"])
    ax.set_ylabel("MAE")
    ax.set_title("MAE per Variable")
    for bar, val in zip(bars, mae_per_var):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5f. 预测散点图（预测值 vs 真实值，温度变量）
    ax = axes[1, 2]
    y_true_flat = Y_test_np[:, :, 0].flatten()
    y_pred_flat = Y_pred_np[:, :, 0].flatten()
    # 降采样以加速渲染
    sample_pts = min(2000, len(y_true_flat))
    idx = np.random.choice(len(y_true_flat), sample_pts, replace=False)
    ax.scatter(y_true_flat[idx], y_pred_flat[idx], alpha=0.3, s=5,
               color="steelblue", edgecolors="none")
    lims = [y_true_flat.min(), y_true_flat.max()]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect Prediction")
    ax.set_xlabel("True Value")
    ax.set_ylabel("Predicted Value")
    ax.set_title(f"Prediction Scatter — Temperature (n={sample_pts})")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle("iTransformer — Multivariate Time Series Forecasting", fontsize=14,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig("F:/note/deep_learning/timeseries/code/04_itransformer_demo.png",
                dpi=150, bbox_inches="tight")
    print("   图表已保存: code/04_itransformer_demo.png")
    plt.close()

    return model


# ============================================================================
# C. 关键函数清单 — 实际项目中你只需要记住这几个 API
# ============================================================================

"""
## iTransformer 常用 API 速查

### 方式一：自己的 iTransformer 类（本文件）

```python
from 04_itransformer_demo import iTransformer, RevIN

model = iTransformer(
    L=96,           # 回溯长度
    H=96,           # 预测长度
    C=7,            # 变量数
    d_model=512,
    n_heads=8,
    e_layers=3,
    d_ff=2048,
    dropout=0.1,
    use_revin=True,
)

# 训练
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for x_batch, y_batch in loader:
    y_hat = model(x_batch)          # (B, H, C)
    loss = F.mse_loss(y_hat, y_batch)
    loss.backward()
    optimizer.step()

# 推理（单步，非自回归）
model.eval()
with torch.no_grad():
    y_pred = model(x_history)       # (B, H, C)
```

### 方式二：官方 TSLib（推荐生产使用）

```python
# git clone https://github.com/thuml/iTransformer.git
# 或使用 Time-Series-Library

from models.iTransformer import Model

model = Model({
    'enc_in': 7,        # 变量数 C
    'seq_len': 96,      # 回溯长度 L
    'pred_len': 96,     # 预测长度 H
    'd_model': 512,
    'n_heads': 8,
    'e_layers': 3,
    'd_ff': 2048,
    'dropout': 0.1,
    'use_norm': 1,      # RevIN
})
```

### 关键超参数调参指南

| 参数     | 默认  | 调参建议                                              |
|---------|------|-----------------------------------------------------|
| L       | —    | 回溯长度，推荐 = H 或 H×2~3。L 太小 token 信息不足               |
| d_model | 512  | 论文默认 512。数据量小→256，数据量大→512/768                    |
| e_layers| 3    | Block 层数。C 小→2，C 大→3~4                              |
| n_heads | 8    | 注意力头数，必须能被 d_model 整除                              |
| d_ff    | 2048 | FFN 隐层维度，通常 2~4× d_model                           |
| dropout | 0.1  | 模型小→0.0，过拟合→0.2                                    |
| use_revin| True | 强烈建议开启，尤其多变量尺度差异大时                                 |

### 适用场景判断

```
开始
 ↓
变量数 C ≥ 3 且变量间有因果关系？
 ├─ 是 → iTransformer ✅
 └─ 否 → 是单变量预测？
            ├─ 是 → DLinear / PatchTST
            └─ 否 → 变量数 C > 100？
                      ├─ 是 → PatchTST (Channel-Independent)
                      └─ 否 → 尝试 iTransformer，与 PatchTST 对比验证
```


### 数据预处理要点

1. **每个变量独立归一化**（z-score 或 RevIN）
   - 不同变量量纲差异极大（温度 20~30，气压 ~1000）
   - RevIN 自动处理，无需手动逐变量归一化

2. **缺失值处理**
   - 缺失率 <5% → 前向填充
   - 缺失率 5%~20% → 线性插值
   - 缺失率 >20% → 考虑丢弃该变量或时间窗口

3. **外生变量 vs 内生变量**
   - 内生变量（预测目标）：C 包含预测目标自身
   - 外生变量（仅做特征）：也放入 C，通过 Attention 自然学习关系
   - iTransformer 不区分内生/外生——反正变量间依赖交给 Attention
"""


# ============================================================================
if __name__ == "__main__":
    train_demo()
