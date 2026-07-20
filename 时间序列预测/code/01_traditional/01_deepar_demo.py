"""
DeepAR 完整学习示例
====================
两套实现，按需选择：

  A. GluonTS (推荐) — Amazon 官方库，开箱即用，生产级
  B. PyTorch 从零构建 — 深入理解内部机制

参考文献：
  Salinas et al. "DeepAR: Probabilistic Forecasting with Autoregressive RNNs" (2017)

环境依赖：
  pip install gluonts torch numpy pandas matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================================
# 0. 合成数据生成（A/B 共用）
# ============================================================================

def make_synthetic_ts(num_series=50, length=300, freq="D", seed=42):
    """生成含趋势+季节性的多条时间序列，模拟跨序列异质性。

    返回 DataFrame，列为 {item_id, timestamp, value}。
    """
    rng = np.random.default_rng(seed)
    records = []

    for i in range(num_series):
        t = np.arange(length)
        trend = 1 + rng.uniform(0.001, 0.005) * t
        seasonal = 1 + rng.uniform(0.1, 0.5) * np.sin(
            2 * np.pi * t / rng.integers(20, 30)
        )
        noise = rng.normal(0, 0.1, length)
        scale = rng.uniform(5, 20)
        values = scale * trend * seasonal + noise

        dates = pd.date_range("2024-01-01", periods=length, freq=freq)
        for d, v in zip(dates, values):
            records.append({"item_id": f"item_{i:03d}", "timestamp": d, "value": v})

    return pd.DataFrame(records)


# ============================================================================
# A. GluonTS — 生产级 DeepAR（推荐）
# ============================================================================

def gluonts_demo():
    """使用 GluonTS 的 DeepAREstimator 完成训练→预测→评估全流程。"""
    print("=" * 55)
    print("A. GluonTS DeepAR — 生产级实现")
    print("=" * 55)

    # ---- 1. 准备数据 ----
    print("\n[1] 构建 GluonTS Dataset...")
    df = make_synthetic_ts(num_series=50, length=300)

    from gluonts.dataset.pandas import PandasDataset
    ds = PandasDataset(df, target="value", item_id="item_id", timestamp="timestamp", freq="D")

    # 按时间切分：前 250 天训练，后 50 天测试
    train_ds = ds.slice_by_time(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-09-06"))
    test_ds = ds.slice_by_time(pd.Timestamp("2024-09-07"), pd.Timestamp("2024-10-26"))
    print(f"   train: {len(train_ds)} series × ~250 steps")
    print(f"   test:  {len(test_ds)} series × ~50 steps")

    # ---- 2. 构建 Estimator ----
    print("\n[2] 构建 DeepAREstimator...")
    from gluonts.torch.model.deepar import DeepAREstimator
    from gluonts.torch.distributions import StudentTOutput

    estimator = DeepAREstimator(
        freq="D",
        prediction_length=28,          # 预测未来 28 天
        context_length=56,             # 用过去 56 天做上下文
        hidden_size=64,                # LSTM 隐藏维度
        num_layers=2,                  # LSTM 层数
        dropout_rate=0.1,
        lr=1e-2,
        batch_size=64,
        num_batches_per_epoch=50,
        epochs=30,
        distr_output=StudentTOutput(),  # ← 厚尾分布，对异常值比高斯更鲁棒
        # distr_output=NegativeBinomialOutput(),  # 计数数据用这个
    )

    # ---- 3. 训练 ----
    print("\n[3] 训练中...")
    predictor = estimator.train(train_ds, cache_data=True)

    # ---- 4. 预测 ----
    print("\n[4] 预测中...")
    from gluonts.evaluation import make_evaluation_predictions

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds, predictor=predictor, num_samples=200
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)
    print(f"   共 {len(forecasts)} 条预测，每条 {forecasts[0].samples.shape[0]} 条采样路径")

    # ---- 5. 评估 ----
    print("\n[5] 评估指标:")
    from gluonts.evaluation import Evaluator

    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(tss, forecasts)
    for k, v in agg_metrics.items():
        print(f"   {k}: {v:.4f}")

    # ---- 6. 可视化其中一条 ----
    fig, ax = plt.subplots(figsize=(12, 4))
    # 画 100 条采样路径 (灰色半透明)
    samples = np.array(forecasts[0].samples)  # (num_samples, pred_len)
    pred_len = samples.shape[1]
    context = tss[0]["target"][-56:]           # 最后 context_length 步

    ax.plot(range(-len(context), 0), context, "k-", linewidth=1.5, label="Context")
    ax.plot(range(pred_len), tss[0]["target"][-pred_len:], "k--", linewidth=1.2, label="True")

    for s in samples[:50]:
        ax.plot(range(pred_len), s, color="blue", alpha=0.03)
    p50 = np.percentile(samples, 50, axis=0)
    p10 = np.percentile(samples, 10, axis=0)
    p90 = np.percentile(samples, 90, axis=0)
    ax.plot(range(pred_len), p50, "b-", linewidth=2, label="P50")
    ax.fill_between(range(pred_len), p10, p90, alpha=0.15, color="blue", label="P10–P90")

    ax.axvline(x=0, color="gray", linestyle=":")
    ax.legend(fontsize=8)
    ax.set_title("GluonTS DeepAR — Forecast Samples & Prediction Intervals")
    fig.tight_layout()
    fig.savefig("F:/note/deep_learning/timeseries/code/01_deepar_gluonts.png", dpi=150)
    print("\n   图表已保存: code/01_deepar_gluonts.png")
    plt.close()


# ============================================================================
# B. PyTorch 从零构建 — 教学用（理解内部机制）
# ============================================================================

def pytorch_demo():
    """用纯 PyTorch 从零构建 DeepAR，深入理解 Teacher Forcing / 祖先采样 / 似然函数。"""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    print("\n" + "=" * 55)
    print("B. PyTorch 从零构建 — 教学用")
    print("=" * 55)

    # ---- 1. 准备数据 ----
    print("\n[1] 准备数据...")
    num_series, series_len = 50, 200
    rng = np.random.default_rng(42)
    data = np.zeros((num_series, series_len))
    for i in range(num_series):
        t = np.arange(series_len)
        trend = 1 + rng.uniform(0.001, 0.005) * t
        seasonal = 1 + rng.uniform(0.1, 0.5) * np.sin(2 * np.pi * t / rng.integers(20, 30))
        noise = rng.normal(0, 0.1, series_len)
        data[i] = rng.uniform(5, 20) * trend * seasonal + noise

    # 规范化每条序列
    scales = np.mean(np.abs(data[:, :100]), axis=1, keepdims=True)
    data_scaled = data / (scales + 1e-6)

    # 训练/测试拆分
    context_len = 30
    pred_len = 20
    train_data = data_scaled[:, :150]
    test_data = data_scaled[:, 130:]

    # 从每条序列采样训练窗口
    windows_past, windows_target = [], []
    for i in range(num_series):
        for _ in range(3):
            start = np.random.randint(0, train_data.shape[1] - context_len - 1)
            end = start + context_len
            windows_past.append(train_data[i, start:end])
            windows_target.append(train_data[i, start + 1 : end + 1])

    loader = DataLoader(
        TensorDataset(
            torch.tensor(np.array(windows_past), dtype=torch.float32),
            torch.tensor(np.array(windows_target), dtype=torch.float32),
        ),
        batch_size=64,
        shuffle=True,
    )

    # ---- 2. 模型定义 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    class DeepAR(nn.Module):
        def __init__(self, hidden=64, num_layers=2, dropout=0.1):
            super().__init__()
            # 输入: 上一步值 (1 维)，输出: μ 和 σ (2 维)
            self.lstm = nn.LSTM(1, hidden, num_layers, dropout=dropout, batch_first=True)
            self.head = nn.Linear(hidden, 2)

        def forward(self, x):
            """Teacher Forcing 训练。x: (B, T) → 返回 (μ, σ)"""
            out, _ = self.lstm(x.unsqueeze(-1))  # (B, T, hidden)
            params = self.head(out)               # (B, T, 2)
            mu = params[..., 0]
            sigma = nn.functional.softplus(params[..., 1]) + 1e-4
            return mu, sigma

        def sample_paths(self, x_last, pred_len, num_samples=200):
            """祖先采样预测。x_last: (B, T) 历史序列"""
            B = x_last.shape[0]
            # 编码历史
            _, (h, c) = self.lstm(x_last.unsqueeze(-1))
            z = x_last[:, -1].clone()
            samples = torch.zeros(num_samples, B, pred_len, device=x_last.device)

            for k in range(num_samples):
                hk, ck, zk = h.clone(), c.clone(), z.clone()
                for t in range(pred_len):
                    _, (hk, ck) = self.lstm(zk.view(B, 1, 1), (hk, ck))
                    p = self.head(hk[-1])  # (B, 2)
                    mu_t = p[:, 0]
                    sigma_t = nn.functional.softplus(p[:, 1]) + 1e-4
                    zk = mu_t + sigma_t * torch.randn_like(sigma_t)
                    samples[k, :, t] = zk
            return samples

    model = DeepAR(hidden=64, num_layers=2, dropout=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ---- 3. 训练 ----
    print("\n[3] 训练...")
    model.train()
    for epoch in range(100):
        total_loss = 0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            mu, sigma = model(x_batch)
            # 负对数似然: NLL = ½log(2π) + log(σ) + (y−μ)²/(2σ²)
            loss = (0.5 * np.log(2 * np.pi) + torch.log(sigma)
                    + 0.5 * ((y_batch - mu) / sigma) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"   Epoch {epoch+1:3d} | Loss: {total_loss/len(loader):.4f}")

    # ---- 4. 预测 + 可视化 ----
    print("\n[4] 预测 + 画图...")
    model.eval()
    x_test = torch.tensor(test_data[0:1, :context_len], dtype=torch.float32).to(device)
    with torch.no_grad():
        samples = model.sample_paths(x_test, pred_len, num_samples=200)

    samples_np = samples[:, 0, :].cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(range(-context_len, 0), test_data[0, :context_len], "k-", linewidth=1.5, label="Context")
    ax.plot(range(pred_len), test_data[0, context_len: context_len + pred_len],
            "k--", linewidth=1.2, label="True")

    p50 = np.percentile(samples_np, 50, axis=0)
    p10 = np.percentile(samples_np, 10, axis=0)
    p90 = np.percentile(samples_np, 90, axis=0)
    ax.plot(range(pred_len), p50, "b-", linewidth=2, label="P50")
    ax.fill_between(range(pred_len), p10, p90, alpha=0.15, color="blue", label="P10–P90")
    ax.axvline(x=0, color="gray", linestyle=":")
    ax.legend(fontsize=8)
    ax.set_title("PyTorch DeepAR from Scratch — Forecast")
    fig.tight_layout()
    fig.savefig("F:/note/deep_learning/timeseries/code/01_deepar_pytorch.png", dpi=150)
    print("   图表已保存: code/01_deepar_pytorch.png")
    plt.close()


# ============================================================================
# C. 关键函数清单 — 实际项目中你只需要记住这几个 API
# ============================================================================

"""
## GluonTS DeepAR 常用 API 速查

```python
from gluonts.dataset.pandas import PandasDataset
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.distributions import StudentTOutput, NegativeBinomialOutput

# 1. 数据准备
ds = PandasDataset(df, target="value", item_id="item_id", timestamp="timestamp", freq="D")
train = ds.slice_by_time(start, end)

# 2. 构建 + 训练（一行）
predictor = DeepAREstimator(
    freq="D",
    prediction_length=28,
    context_length=56,
    hidden_size=64,
    num_layers=2,
    distr_output=StudentTOutput(),   # 连续值
    # distr_output=NegativeBinomialOutput(),  # 计数
).train(train)

# 3. 预测 + 评估
from gluonts.evaluation import make_evaluation_predictions, Evaluator
forecast_it, ts_it = make_evaluation_predictions(test, predictor, num_samples=200)
forecasts = list(forecast_it)

# 4. 取分位数
samples = np.array(forecasts[0].samples)
p50 = np.percentile(samples, 50, axis=0)
p10, p90 = np.percentile(samples, 10, axis=0), np.percentile(samples, 90, axis=0)
```

## 关键超参数调参指南

| 参数 | 默认 | 调参建议 |
|------|------|---------|
| context_length | — | 预测长度的 2~5 倍 |
| hidden_size | 64 | 数据量大→128，小→32 |
| num_layers | 2 | 序列长→3，短→1 |
| distr_output | StudentT | 连续值默认 StudentT，计数用 NegBin |
| epochs | 100 | 配合 early stopping 使用 |
| dropout_rate | 0.1 | 过拟合→0.2~0.3 |

## 分布选择决策树

- 连续值 + 可能有异常值   → StudentTOutput()
- 连续值 + 无异常值        → GaussianOutput()
- 非负整数（销量/访问量）  → NegativeBinomialOutput()
- (0,1) 区间               → BetaOutput()
"""


# ============================================================================
if __name__ == "__main__":
    gluonts_demo()    # ← 生产推荐
    # pytorch_demo()  # ← 教学用（需手动取消注释）
