"""
TCN (Temporal Convolutional Network) 完整学习示例
==================================================
两套实现，按需选择：

  A. tsai (推荐) — 时序深度学习专用库，TCN 开箱即用
  B. PyTorch 从零构建 — 深入理解因果膨胀卷积 + 残差块

参考文献：
  Bai et al. "An Empirical Evaluation of Generic Convolutional and
  Recurrent Networks for Sequence Modeling" (2018)

环境依赖：
  pip install tsai torch numpy pandas matplotlib scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================================
# 0. 合成数据生成（A/B 共用）
# ============================================================================

def make_synthetic_ts(num_series=10, length=2000, freq="h", seed=42):
    """生成含日周期+周周期的多变量时间序列，模拟电力负荷场景。

    返回 (data_3d, scaler_params)。
    data_3d: (num_series, length, num_features) — 最后维度包含 target + 协变量。
    """
    rng = np.random.default_rng(seed)
    t = np.arange(length)

    # 日周期 (24h) 和 周周期 (168h)
    daily = np.sin(2 * np.pi * t / 24)
    weekly = 0.5 * np.sin(2 * np.pi * t / 168)

    series_list = []
    for i in range(num_series):
        base_load = rng.uniform(50, 150)
        trend = 1 + rng.uniform(-0.0001, 0.0002) * t
        noise = rng.normal(0, base_load * 0.05, length)

        load = base_load * trend * (1 + 0.3 * daily + 0.15 * weekly) + noise
        load = np.maximum(load, 0)  # 负荷不能为负

        # 合成协变量
        hour_of_day = t % 24
        day_of_week = (t // 24) % 7
        temp = 20 + 5 * daily + rng.normal(0, 2, length)  # 模拟温度

        features = np.column_stack([load, hour_of_day / 24, day_of_week / 7, temp / 40])
        series_list.append(features)

    data = np.stack(series_list, axis=0)  # (num_series, T, 4)

    # 逐序列 target 归一化参数（只对 load 列）
    load_col = data[:, :, 0]
    mean = load_col.mean(axis=1, keepdims=True)
    std = load_col.std(axis=1, keepdims=True) + 1e-6

    return data, (mean, std)


def make_sliding_windows(data_3d, input_len, pred_len, stride=1):
    """从 (N, T, F) 数据生成 (X, y) 滑动窗口。

    X: (num_windows, F, input_len)   — Conv1d 格式
    y: (num_windows, pred_len)       — 只预测 target (第一列)
    """
    X_list, y_list = [], []
    N, T, F = data_3d.shape

    for i in range(N):
        seq = data_3d[i]
        for start in range(0, T - input_len - pred_len + 1, stride):
            end = start + input_len
            X_list.append(seq[start:end].T)              # (F, input_len)
            y_list.append(seq[end:end + pred_len, 0])    # (pred_len,)

    return np.stack(X_list), np.stack(y_list)


# ============================================================================
# A. tsai — 生产级 TCN（推荐）
# ============================================================================

def tsai_demo():
    """使用 tsai 库的 TCN 完成训练→预测→评估全流程。"""
    print("=" * 55)
    print("A. tsai TCN — 生产级实现")
    print("=" * 55)

    # ---- 1. 准备数据 ----
    print("\n[1] 构建 tsai 数据集...")
    data_3d, (mean, std) = make_synthetic_ts(num_series=5, length=2000)

    input_len = 168   # 过去 7 天 (168 小时)
    pred_len = 24     # 预测未来 1 天 (24 小时)
    num_features = data_3d.shape[2]

    X, y = make_sliding_windows(data_3d, input_len, pred_len, stride=12)
    print(f"   X: {X.shape}  y: {y.shape}")
    # X: (samples, F, input_len), y: (samples, pred_len)

    # Train/val/test split (chronological — 不能用 shuffle split!)
    n_total = len(X)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.85)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_val], y[n_train:n_val]
    X_test, y_test = X[n_val:], y[n_val:]
    print(f"   train: {X_train.shape[0]} | val: {X_val.shape[0]} | test: {X_test.shape[0]}")

    import torch

    # ---- 2. 构建 TCN 模型 ----
    print("\n[2] 构建 TCN 模型...")
    from tsai.models.TCN import TCN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TCN(
        c_in=num_features,       # 输入通道 = 特征数
        c_out=pred_len,          # 输出 = 预测长度
        layers=[64, 64, 64, 64, 64, 64],  # 6 层，每层 64 通道
        ks=3,                    # kernel_size
        fc_dropout=0.1,
    ).to(device)
    print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   感受野: ~{1 + 2 * (3-1) * (2**6 - 1)} 步")

    # ---- 3. 训练 ----
    print("\n[3] 训练...")
    from torch.utils.data import DataLoader, TensorDataset

    train_set = TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    )
    val_set = TensorDataset(
        torch.FloatTensor(X_val), torch.FloatTensor(y_val)
    )

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = torch.nn.MSELoss()

    best_val_loss = float("inf")
    patience, patience_counter = 5, 0

    for epoch in range(50):
        # --- train ---
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)  # (B, pred_len)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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

        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_state)
                break

    # ---- 4. 预测 + 可视化 ----
    print("\n[4] 预测 + 画图...")
    model.eval()
    x_test_tensor = torch.FloatTensor(X_test[-1:]).to(device)
    with torch.no_grad():
        pred = model(x_test_tensor).cpu().numpy()[0]  # (pred_len,)

    # 反归一化
    s_idx = n_val  # 这条序列在原始数据中的起始 index
    # 取测试集最后一条对应的 true 值
    y_test_true = y_test[-1]           # 归一化后的真实值

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    # 左图：最后一条样本的预测 vs 真实
    context = X_test[-1, 0, -168:]  # load 通道, 最近 168 步
    ax1.plot(range(-168, 0), context, "k-", linewidth=1, alpha=0.7, label="History")
    ax1.plot(range(pred_len), y_test_true, "k--", linewidth=1.5, label="True")
    ax1.plot(range(pred_len), pred, "r-", linewidth=1.5, label="TCN Predict")
    ax1.axvline(x=0, color="gray", linestyle=":")
    ax1.legend(fontsize=8)
    ax1.set_title("TCN — Single Sample Forecast")
    ax1.set_xlabel("Hours")

    # 右图：测试集整体 MAE 分布
    with torch.no_grad():
        all_preds = model(torch.FloatTensor(X_test).to(device)).cpu().numpy()
    mae = np.mean(np.abs(all_preds - y_test), axis=1)
    ax2.hist(mae, bins=30, color="steelblue", edgecolor="white")
    ax2.axvline(np.median(mae), color="red", linestyle="--", label=f"Median MAE={np.median(mae):.3f}")
    ax2.set_title("TCN — Test MAE Distribution")
    ax2.set_xlabel("MAE (normalized load)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("F:/note/deep_learning/timeseries/code/02_tcn_tsai.png", dpi=150)
    print("   图表已保存: code/02_tcn_tsai.png")
    plt.close()


# ============================================================================
# B. PyTorch 从零构建 — 教学用（理解内部机制）
# ============================================================================

def pytorch_demo():
    """用纯 PyTorch 从零构建 TCN，深入理解因果膨胀卷积 + 残差块。"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    print("\n" + "=" * 55)
    print("B. PyTorch 从零构建 — 教学用")
    print("=" * 55)

    # ---- 1. 准备数据 ----
    print("\n[1] 准备数据...")
    data_3d, (mean, std) = make_synthetic_ts(num_series=3, length=1500)

    input_len = 168
    pred_len = 24
    num_features = data_3d.shape[2]

    X, y = make_sliding_windows(data_3d, input_len, pred_len, stride=6)
    n_train = int(len(X) * 0.8)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    print(f"   X: {X.shape}, y: {y.shape}")
    print(f"   train: {X_train.shape[0]} | test: {X_test.shape[0]}")

    # ---- 2. TCN 从零构建 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    class CausalConv1d(nn.Module):
        """因果膨胀卷积：只在左侧 padding，保证 t 时刻看不到 > t 的信息。"""
        def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
            super().__init__()
            self.padding = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(
                in_channels, out_channels, kernel_size,
                dilation=dilation, padding=0,
            )
            self.weight_norm = nn.utils.weight_norm(self.conv)

        def forward(self, x):
            # x: (B, C_in, T)
            x = F.pad(x, (self.padding, 0))  # 只在左边 pad
            return self.conv(x)

    class TCNResidualBlock(nn.Module):
        """TCN 残差块：两层膨胀因果卷积 + WeightNorm + ReLU + Dropout + 残差连接。

        ┌─────────────────────────────────┐
        │  CausalConv (in → out, d)       │
        │  WeightNorm → ReLU → Dropout    │
        │  CausalConv (out → out, d)      │
        │  WeightNorm → ReLU → Dropout    │
        │       +                          │
        │  1×1 Conv (if in ≠ out)         │
        │       = ReLU                     │
        └─────────────────────────────────┘
        """
        def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
            super().__init__()
            self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
            self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
            self.dropout = nn.Dropout(dropout)

            # 残差连接 — 通道数不匹配时用 1×1 Conv 对齐
            self.residual = (
                nn.Conv1d(in_channels, out_channels, 1)
                if in_channels != out_channels
                else nn.Identity()
            )

        def forward(self, x):
            # 主路径
            out = self.conv1(x)
            out = F.relu(out)
            out = self.dropout(out)

            out = self.conv2(out)
            out = F.relu(out)
            out = self.dropout(out)

            # 残差
            res = self.residual(x)
            return F.relu(out + res)

    class TCN(nn.Module):
        """Temporal Convolutional Network — 完整实现。

        Parameters
        ----------
        c_in : int         输入通道数（特征维度）
        c_out : int        输出维度（预测长度）
        channels : list    每层输出通道，如 [64]*6
        kernel_size : int  卷积核大小
        dropout : float    每个残差块内的 dropout 概率
        """
        def __init__(self, c_in, c_out, channels, kernel_size=3, dropout=0.1):
            super().__init__()
            layers = []
            num_layers = len(channels)

            for i in range(num_layers):
                dilation = 2 ** i  # dilation: 1, 2, 4, 8, 16, 32, ...
                in_ch = c_in if i == 0 else channels[i - 1]
                out_ch = channels[i]
                layers.append(
                    TCNResidualBlock(in_ch, out_ch, kernel_size, dilation, dropout)
                )

            self.network = nn.Sequential(*layers)
            # 输出投影：取最后 pred_len 步的输出 → 线性投影到预测值
            self.output_proj = nn.Conv1d(channels[-1], 1, kernel_size=1)

            # 计算感受野
            self.receptive_field = 1 + 2 * (kernel_size - 1) * (2 ** num_layers - 1)

        def forward(self, x):
            """x: (B, C_in, T_in) → output: (B, T_out)"""
            # TCN 主干
            out = self.network(x)                      # (B, C_last, T_in)
            # 只取最后 pred_len 步的输出做预测
            pred_len = 24  # 应与构造参数一致
            out = out[:, :, -pred_len:]                # (B, C_last, pred_len)
            out = self.output_proj(out).squeeze(1)     # (B, pred_len)
            return out

        def compute_receptive_field(self):
            return self.receptive_field

    # ---- 3. 实例化模型 ----
    num_layers = 6
    channels = [64] * num_layers
    model = TCN(
        c_in=num_features,
        c_out=pred_len,
        channels=channels,
        kernel_size=3,
        dropout=0.1,
    ).to(device)

    rf = model.compute_receptive_field()
    print(f"   层数: {num_layers}, dilations: {[2**i for i in range(num_layers)]}")
    print(f"   感受野: {rf} 步 (输入窗口: {input_len})")
    print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ---- 4. 训练 ----
    print("\n[3] 训练...")
    train_set = TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    )
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(40):
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1:3d} | Loss: {total_loss/len(train_loader):.4f}")

    # ---- 5. 预测 + 可视化 ----
    print("\n[4] 预测 + 画图...")
    model.eval()
    x_test_tensor = torch.FloatTensor(X_test[:1]).to(device)
    with torch.no_grad():
        pred = model(x_test_tensor).cpu().numpy()[0]

    y_test_true = y_test[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    # 左图：预测 vs 真实
    context = X_test[0, 0, -168:]
    ax1.plot(range(-168, 0), context, "k-", linewidth=1, alpha=0.7, label="History")
    ax1.plot(range(pred_len), y_test_true, "k--", linewidth=1.5, label="True")
    ax1.plot(range(pred_len), pred, "r-", linewidth=1.5, label="TCN (from scratch)")
    ax1.axvline(x=0, color="gray", linestyle=":")
    ax1.legend(fontsize=8)
    ax1.set_title("PyTorch TCN from Scratch — Forecast")
    ax1.set_xlabel("Hours")

    # 右图：感受野结构示意
    ax2.barh(
        [f"d={2**i}" for i in range(num_layers)],
        [2 ** i for i in range(num_layers)],
        color="steelblue", edgecolor="white",
    )
    ax2.set_xscale("log")
    ax2.set_title(f"TCN Dilation Schedule (RF={rf})")
    ax2.set_xlabel("Dilation Factor (log scale)")

    fig.tight_layout()
    fig.savefig("F:/note/deep_learning/timeseries/code/02_tcn_pytorch.png", dpi=150)
    print("   图表已保存: code/02_tcn_pytorch.png")
    plt.close()


# ============================================================================
# C. 关键函数清单 — 实际项目中你只需要记住这几个 API
# ============================================================================

"""
## tsai TCN 常用 API 速查

```python
from tsai.models.TCN import TCN

# 1. 构建模型
model = TCN(
    c_in=5,                # 输入特征数
    c_out=24,              # 输出长度（预测步数）
    layers=[64, 64, 64, 64, 64, 64],  # 每层通道数
    ks=3,                  # kernel_size
    fc_dropout=0.1,        # dropout
)

# 2. 数据格式 (Conv1d / tsai 通用格式)
# X: (batch, channels, time)  ← 注意: 时间在最后!
# y: (batch, prediction_length)

# 3. 训练 (标准 PyTorch)
pred = model(X)            # (B, T_out) — 直接输出预测序列
loss = F.mse_loss(pred, y)

# 4. 感受野计算
def tcn_receptive_field(k, L):
    return 1 + 2 * (k - 1) * (2 ** L - 1)
# k=3, L=6 → RF = 253
# k=3, L=8 → RF = 1021
```

## PyTorch 从零核心组件 (part B)

```python
# 因果膨胀卷积 — TCN 的核心原语
class CausalConv1d(nn.Module):
    def __init__(self, in_c, out_c, k, d=1):
        super().__init__()
        self.pad = (k - 1) * d
        self.conv = nn.Conv1d(in_c, out_c, k, dilation=d)
        nn.utils.weight_norm(self.conv)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))

# 残差块 — TCN 的基本单元
class TCNResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, k, d, dropout):
        self.conv1 = CausalConv1d(in_c, out_c, k, d)
        self.conv2 = CausalConv1d(out_c, out_c, k, d)
        self.residual = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        out = F.relu(self.conv2(F.relu(self.conv1(x))))
        return F.relu(out + self.residual(x))
```

## 关键超参数调参指南

| 参数 | 默认 | 调参建议 |
|------|------|---------|
| kernel_size | 3 | 试 [2, 3, 5, 7]。小=精细局部模式，大=粗粒度模式 |
| num_layers (L) | 6 | RF = 1 + 2(k-1)(2^L-1) ≥ 输入长度。一般 6~8 |
| channels | [64]*L | 数据和特征多→128，少→32。也可 [64, 64, 128, 128] |
| dropout | 0.1 | 过拟合→0.2~0.3。可在各个 block 内不同 |
| learning_rate | 1e-3 | Adam 默认，TCN 训练稳定，可尝试 [5e-4, 2e-3] |
| weight_decay | 1e-4 | L2 正则，[1e-5, 1e-3] |
| batch_size | 64 | 显存允许尽量大，TCN 并行效率随 batch 增大显著提升 |

## 感受野速查表 (k=3)

| L | dilations | RF | 适合的输入窗口 |
|---|-----------|-----|--------------|
| 4 | [1,2,4,8] | 61 | ≤ 60 |
| 5 | [1,2,4,8,16] | 125 | ≤ 125 |
| 6 | [1,2,4,8,16,32] | 253 | ≤ 250 |
| 7 | [1,2,4,8,16,32,64] | 509 | ≤ 500 |
| 8 | [1,2,4,8,16,32,64,128] | 1021 | ≤ 1000 |

## 模型选择决策树

- 需要概率输出 (分位数/分布) → DeepAR / MQ-RNN
- 需要高速推理 (在线/边缘) → TCN ✅
- 序列 > 1000 步 + 需要最好的精度 → PatchTST / iTransformer
- 数据量少 (< 1k 序列) → TCN (不如 Transformer 容易过拟合)
- 只做简单 baseline → DLinear / 朴素方法
- Ensemble 组件 → TCN + LightGBM + DeepAR (多样化 = 鲁棒)

## 常见坑

1. ⚠️ 数据格式：Conv1d 要 (B, C, T)，不是 (B, T, C)，记得 transpose!
2. ⚠️ 感受野不够：RF 必须 ≥ 输入窗口长度，否则"近视"
3. ⚠️ 时序数据打乱：train/val/test 必须按时间切分，不能用 random split!
4. ⚠️ 因果性：做分类/异常检测时可以取消因果约束（双向 pad），做预测时必须是因果的。
5. ⚠️ 归一化泄漏：先切分 train/val/test，再在 train 上 fit scaler，避免信息泄露。
"""


# ============================================================================
if __name__ == "__main__":
    tsai_demo()      # ← 生产推荐
    # pytorch_demo()  # ← 教学用（需手动取消注释）
