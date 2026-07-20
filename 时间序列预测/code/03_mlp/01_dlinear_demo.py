"""
DLinear 完整学习示例
====================
DLinear 极简到不需要第三方时序库——纯 PyTorch 从零构建，完整代码 < 80 行。

参考文献：
  Zeng et al. "Are Transformers Effective for Time Series Forecasting?" (AAAI 2023)

环境依赖：
  pip install torch numpy pandas matplotlib scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ============================================================================
# 0. 合成数据生成
# ============================================================================

def make_synthetic_ts(num_series=10, length=2000, freq="h", seed=42):
    """生成含日周期+周周期的多变量时间序列。

    返回 (data_3d, scaler_params).
    data_3d: (num_series, length, num_channels) — 多变量时序.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(length)

    series_list = []
    for i in range(num_series):
        # 基础周期
        daily = np.sin(2 * np.pi * t / 24)
        weekly = 0.5 * np.sin(2 * np.pi * t / 168)

        base = rng.uniform(50, 150)
        trend = 1 + rng.uniform(-0.0001, 0.0002) * t
        noise = rng.normal(0, base * 0.08, length)

        # Channel 0: 主序列 (负荷)
        load = base * trend * (1 + 0.3 * daily + 0.15 * weekly) + noise

        # Channel 1: 协变量 (温度, 和负荷有负相关)
        temp = 20 + 8 * daily + rng.normal(0, 3, length)

        # Channel 2: 协变量 (湿度)
        humidity = 60 + 20 * np.sin(2 * np.pi * t / (24 * 3 + 7)) + rng.normal(0, 10, length)

        features = np.column_stack([
            load,                       # (length,)
            temp / 50,                  # 0~1 区间
            humidity / 100,             # 0~1 区间
        ])
        series_list.append(features)

    data = np.stack(series_list, axis=0)  # (num_series, T, C)

    # 逐序列归一化 target (第 0 列)
    target = data[:, :, 0]
    mean = target.mean(axis=1, keepdims=True)
    std = target.std(axis=1, keepdims=True) + 1e-6

    return data, (mean, std)


def make_sliding_windows(data_3d, L, H, stride=1):
    """滑动窗口: (N, T, C) → X (N_w, L, C), y (N_w, H, C)."""
    X_list, y_list = [], []
    N, T, C = data_3d.shape

    for i in range(N):
        seq = data_3d[i]
        for start in range(0, T - L - H + 1, stride):
            X_list.append(seq[start:start + L])       # (L, C)
            y_list.append(seq[start + L:start + L + H])  # (H, C)

    return np.stack(X_list), np.stack(y_list)


# ============================================================================
# A. DLinear 核心实现（从零构建，< 40 行）
# ============================================================================

class MovingAvg(nn.Module):
    """沿时间维度做移动平均，提取趋势分量。"""

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        # x: (B, L, C) → permute → (B, C, L) → avg_pool1d → permute → (B, L, C)
        x = x.permute(0, 2, 1)                        # (B, C, L)
        x = F.avg_pool1d(
            x, self.kernel_size, stride=1,
            padding=self.kernel_size // 2,
        )                                              # (B, C, L)
        return x.permute(0, 2, 1)                     # (B, L, C)


class DLinear(nn.Module):
    """Decomposition Linear — 一个移动平均 + 两个 Linear 层。

    Parameters
    ----------
    L : int           输入/历史窗口长度 (lookback)
    H : int           预测长度 (horizon)
    C : int           变量/通道数
    kernel_size : int 移动平均的窗口大小 (默认 25)
    """
    def __init__(self, L, H, C, kernel_size=25):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)
        # 两个独立的线性层，分别处理趋势和残差
        # nn.Linear(L, H): 将最后维 L → H, 接收 (*, L) → (*, H)
        self.Linear_Trend = nn.Linear(L, H)
        self.Linear_Rem   = nn.Linear(L, H)

    def forward(self, x):
        """x: (B, L, C) → y_hat: (B, H, C)."""
        # 1. Decomposition
        trend = self.moving_avg(x)       # (B, L, C) — 低频趋势
        rem   = x - trend                # (B, L, C) — 高频残差

        # 2. 转换为 (*, L) 格式让 nn.Linear 沿时间维度工作
        trend = trend.permute(0, 2, 1)   # (B, C, L)
        rem   = rem.permute(0, 2, 1)     # (B, C, L)

        # 3. 每个通道独立映射 L → H
        trend_pred = self.Linear_Trend(trend)  # (B, C, H)
        rem_pred   = self.Linear_Rem(rem)      # (B, C, H)

        # 4. 加回, 恢复 (B, H, C) 格式
        return (trend_pred + rem_pred).permute(0, 2, 1)   # (B, H, C)


# ============================================================================
# B. NLinear — DLinear 的"去分解版" (论文中也提出，作为对照)
# ============================================================================

class NLinear(nn.Module):
    """纯 Linear baseline: 输入减最后一步值（归一化） + 一层 Linear。

    比 DLinear 更简单——去掉了移动平均分解，只做一次减均值的归一化。
    在部分 benchmark 上和 DLinear 性能相当，说明分解有时不是必需的。
    """
    def __init__(self, L, H, C):
        super().__init__()
        self.Linear = nn.Linear(L, H)

    def forward(self, x):
        # x: (B, L, C)
        last = x[:, -1:, :]                      # (B, 1, C) — 最后一步值
        x_norm = x - last                         # (B, L, C) — 减去最后一步
        x_norm = x_norm.permute(0, 2, 1)          # (B, C, L)
        out = self.Linear(x_norm)                 # (B, C, H)
        out = out.permute(0, 2, 1)                # (B, H, C)
        return out + last.permute(0, 2, 1)        # 加回减去的值


# ============================================================================
# C. 训练 + 评估 + 可视化
# ============================================================================

def train_and_eval(model, train_loader, val_loader, epochs=30, lr=1e-3,
                   device="cpu", early_stop_patience=5):
    """标准训练循环：MSE loss + Adam + Cosine 退火 + Early Stopping。"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        # --- train ---
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # --- val ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item()
        val_loss /= len(val_loader)
        scheduler.step()

        # --- early stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"   Early stopping @ epoch {epoch+1}")
                model.load_state_dict(best_state)
                break

        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    return model


def demo():
    print("=" * 55)
    print("DLinear — Decomposition Linear Forecasting")
    print("=" * 55)

    # ---- 1. 准备数据 ----
    print("\n[1] 准备数据...")
    data_3d, (mean, std) = make_synthetic_ts(num_series=8, length=2000)
    N, T, C = data_3d.shape
    print(f"   {N} 条序列 × {T} 步 × {C} 变量")

    L, H = 168, 24          # 过去 7 天 → 未来 1 天
    kernel_size = 25

    X, y = make_sliding_windows(data_3d, L, H, stride=12)
    print(f"   X: {X.shape}  y: {y.shape}")

    # Chronological split
    n_total = len(X)
    n_train = int(n_total * 0.7)
    n_val   = int(n_total * 0.85)

    X_train, y_train = torch.FloatTensor(X[:n_train]), torch.FloatTensor(y[:n_train])
    X_val,   y_val   = torch.FloatTensor(X[n_train:n_val]), torch.FloatTensor(y[n_train:n_val])
    X_test,  y_test  = torch.FloatTensor(X[n_val:]),       torch.FloatTensor(y[n_val:])

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=64, shuffle=False)

    print(f"   train: {len(X_train)} | val: {len(X_val)} | test: {len(X_test)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 2. 构建 DLinear ----
    print(f"\n[2] 构建 DLinear (L={L}, H={H}, C={C}, kernel_size={kernel_size})...")
    model = DLinear(L=L, H=H, C=C, kernel_size=kernel_size)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   参数量: {n_params:,}")
    print(f"   对比: Informer ~1M params → DLinear 只用 {n_params/1e6*100:.2f}%")

    # 感受一下 W 矩阵大小
    w = model.Linear_Trend.weight.data  # (H, L)
    print(f"   W_trend 形状: {tuple(w.shape)} — 每个预测步 ({H}) 对 {L} 个历史步各有一个权重")

    # ---- 3. 训练 ----
    print(f"\n[3] 训练 (device={device})...")
    model = train_and_eval(model, train_loader, val_loader, epochs=30,
                           device=device, early_stop_patience=5)

    # ---- 4. 评估 ----
    print("\n[4] 评估...")
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test.to(device)).cpu().numpy()
    test_true = y_test.numpy()

    # target channel (channel 0) 的 MSE / MAE
    mse = np.mean((test_pred[:, :, 0] - test_true[:, :, 0]) ** 2)
    mae = np.mean(np.abs(test_pred[:, :, 0] - test_true[:, :, 0]))
    print(f"   Test MSE (ch0): {mse:.4f}")
    print(f"   Test MAE (ch0): {mae:.4f}")

    # ---- 5. 可视化 ----
    print("\n[5] 生成图表...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) 预测 vs 真实 (最后一条测试样本, channel 0)
    ax = axes[0, 0]
    last_sample_pred = test_pred[-1, :, 0]
    last_sample_true = test_true[-1, :, 0]
    context = X_test[-1, -min(L, 168):, 0].numpy()
    ax.plot(range(-len(context), 0), context, "k-", linewidth=1, alpha=0.7, label="History")
    ax.plot(range(H), last_sample_true, "k--", linewidth=1.5, label="True")
    ax.plot(range(H), last_sample_pred, "r-", linewidth=1.5, label="DLinear")
    ax.axvline(x=0, color="gray", linestyle=":")
    ax.legend(fontsize=8)
    ax.set_title(f"DLinear Forecast (L={L}, H={H})")
    ax.set_xlabel("Hours")

    # (b) W_trend 热力图 — 模型的可解释性核心
    ax = axes[0, 1]
    w_trend = model.Linear_Trend.weight.data.cpu().numpy()  # (H, L)
    im = ax.imshow(w_trend, aspect="auto", cmap="RdBu_r", vmin=-0.1, vmax=0.1)
    ax.set_title("W_trend Heatmap — What history matters?")
    ax.set_xlabel("History step (L)")
    ax.set_ylabel("Prediction step (H)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # (c) MAE per prediction step
    ax = axes[1, 0]
    mae_per_step = np.mean(np.abs(test_pred[:, :, 0] - test_true[:, :, 0]), axis=0)
    ax.bar(range(H), mae_per_step, color="steelblue", edgecolor="white")
    ax.axhline(mae, color="red", linestyle="--", label=f"Avg MAE={mae:.3f}")
    ax.set_title("MAE per Prediction Step")
    ax.set_xlabel("Future hour")
    ax.set_ylabel("MAE")
    ax.legend(fontsize=8)

    # (d) kernel_size 对比实验 (快速 scan)
    ax = axes[1, 1]
    ks_list = [5, 15, 25, 51, 101]
    mse_list = []
    print("   Running kernel_size ablation (each takes ~5s)...")
    for ks in ks_list:
        m = DLinear(L=L, H=H, C=C, kernel_size=ks).to(device)
        # 只跑 10 epochs 快速对比
        m = train_and_eval(m, train_loader, val_loader, epochs=10,
                           device=device, early_stop_patience=3)
        m.eval()
        with torch.no_grad():
            p = m(X_val.to(device)).cpu().numpy()
        mse_ks = np.mean((p[:, :, 0] - y_val.numpy()[:, :, 0]) ** 2)
        mse_list.append(mse_ks)
        print(f"   kernel_size={ks:3d}  val MSE={mse_ks:.4f}")

    ax.plot(ks_list, mse_list, "o-", color="steelblue", markersize=8)
    ax.set_xlabel("kernel_size")
    ax.set_ylabel("Val MSE (ch0)")
    ax.set_title("Impact of kernel_size")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("F:/note/deep_learning/timeseries/code/03_dlinear_demo.png", dpi=150)
    print("   图表已保存: code/03_dlinear_demo.png")
    plt.close()

    # ---- 6. NLinear 对比 ----
    print("\n[6] NLinear (无分解版) 对比...")
    nlinear = NLinear(L=L, H=H, C=C).to(device)
    nlinear = train_and_eval(nlinear, train_loader, val_loader, epochs=30,
                             device=device, early_stop_patience=5)
    nlinear.eval()
    with torch.no_grad():
        nl_pred = nlinear(X_test.to(device)).cpu().numpy()
    nl_mse = np.mean((nl_pred[:, :, 0] - test_true[:, :, 0]) ** 2)
    nl_mae = np.mean(np.abs(nl_pred[:, :, 0] - test_true[:, :, 0]))
    print(f"   NLinear Test MSE: {nl_mse:.4f}, MAE: {nl_mae:.4f}")
    print(f"   DLinear Test MSE: {mse:.4f}, MAE: {mae:.4f}")
    print(f"   Improvement from decomposition: {(nl_mse - mse)/nl_mse*100:.1f}%")


# ============================================================================
# D. 关键 API 速查 + 超参指南
# ============================================================================

"""
## DLinear 核心 API (纯 PyTorch, < 40 行)

```python
import torch.nn as nn
import torch.nn.functional as F

class DLinear(nn.Module):
    def __init__(self, L, H, C, kernel_size=25):
        super().__init__()
        self.Linear_Trend = nn.Linear(L, H)
        self.Linear_Rem   = nn.Linear(L, H)
        self.kernel_size = kernel_size

    def moving_avg(self, x):
        # x: (B, L, C)
        x = x.permute(0, 2, 1)
        x = F.avg_pool1d(x, self.kernel_size, stride=1,
                         padding=self.kernel_size // 2)
        return x.permute(0, 2, 1)

    def forward(self, x):
        trend = self.moving_avg(x)
        rem = x - trend
        trend = trend.permute(0, 2, 1)
        rem   = rem.permute(0, 2, 1)
        out = self.Linear_Trend(trend) + self.Linear_Rem(rem)
        return out.permute(0, 2, 1)
```

## 关键超参数

| 参数 | 默认 | 说明 |
|------|------|------|
| L (lookback) | 336 | 和预测长度成比例, 一般 L ≥ 2*H |
| H (horizon) | 96 | 预测窗口, DLinear 擅长 96~720 |
| kernel_size | 25 | 唯一需调的超参。试 [15, 25, 51, 101] |
| lr | 1e-3 | Adam, 几乎不需要调 |
| batch_size | 32 | 影响不大, 模型太小 |

## 常见坑

1. ⚠️ DLinear 是确定性点预测, 无法输出概率/分位数
2. ⚠️ 时序切分必须是 chronological split, 不能 random shuffle
3. ⚠️ 预测窗口 H 太长时参数量线性增长: W_trend 有 L*H 个参数
4. ⚠️ kernel_size=1 等于不做分解, DLinear 退化为纯 Linear × 2
5. ⚠️ 如果 test 上 DLinear < Naive baseline, 检查数据是否被污染

## DLinear 的三层理解

1. **表层**: 移动平均 → 两个 Linear → 相加
2. **中层**: 把时序分解成"慢变量"和"快变量", 各用专用模型预测
3. **深层**: "简单模型优先"的方法论——任何复杂模型都必须先证明自己比 Linear 更好

## 模型对比速查

| | DLinear | TCN | DeepAR |
|---|---|---|---|
| 参数 | ~5K | ~100K | ~100K |
| 训练 | 极快 | 快 | 慢 |
| 推理 | 极快 | 快 | 中等 |
| 概率 | ❌ | 需 DIY | ✅ 原生 |
| 可解释 | ⭐⭐⭐⭐⭐ W矩阵 | ⭐⭐⭐ 感受野 | ⭐⭐ 隐状态 |
| 非线性 | ❌ | ✅ | ✅ |
"""


# ============================================================================
if __name__ == "__main__":
    demo()
