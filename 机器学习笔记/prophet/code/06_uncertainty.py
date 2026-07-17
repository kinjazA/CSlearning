"""
第 7 章：不确定性量化
======================
- MAP 默认模式的不确定性
- uncertainty_samples 对比
- MCMC 采样（备选）
- 预测区间解读与验证
- 区间覆盖率检查
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import time

# ============================================================
# 0. 数据准备
# ============================================================
url = "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_air_passengers.csv"
df = pd.read_csv(url)
print(f"数据量: {len(df)}")

# ============================================================
# 1. 默认 MAP 模式 — 查看趋势不确定性
# ============================================================
model = Prophet(
    interval_width=0.80,
    uncertainty_samples=1000,
)
model.fit(df)
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# 可视化
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 预测 + 区间
ax = axes[0]
model.plot(forecast, ax=ax)
ax.set_title('预测 + 80% 预测区间')

# 趋势 + 趋势区间
ax = axes[1]
ax.plot(forecast['ds'], forecast['trend'], 'b-', label='Trend')
ax.fill_between(
    forecast['ds'],
    forecast['trend_lower'],
    forecast['trend_upper'],
    alpha=0.2, color='blue', label='Trend 80% interval'
)
ax.set_title('趋势 + 趋势不确定性')
ax.legend()

plt.tight_layout()
plt.show()

# ============================================================
# 2. uncertainty_samples 对比
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, samples in zip(axes, [10, 100, 1000]):
    m = Prophet(uncertainty_samples=samples, yearly_seasonality=False)
    m.fit(df)
    fc = m.predict(m.make_future_dataframe(90))
    ax.plot(fc['ds'], fc['yhat'], 'b-', lw=1)
    ax.fill_between(fc['ds'], fc['yhat_lower'], fc['yhat_upper'], alpha=0.2)
    ax.fill_between(fc['ds'], fc['trend_lower'], fc['trend_upper'], alpha=0.2, color='red')
    ax.set_title(f'uncertainty_samples = {samples}\n(蓝=预测区间, 红=趋势区间)')

fig.suptitle('uncertainty_samples 对区间平滑度的影响', fontsize=14)
plt.tight_layout()
plt.show()

# ============================================================
# 3. 不同 interval_width 对比
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, iw in zip(axes, [0.50, 0.80, 0.95]):
    m = Prophet(interval_width=iw, yearly_seasonality=False)
    m.fit(df)
    fc = m.predict(m.make_future_dataframe(90))
    ax.plot(fc['ds'], fc['yhat'], 'b-', lw=1)
    ax.fill_between(fc['ds'], fc['yhat_lower'], fc['yhat_upper'], alpha=0.2)
    ax.set_title(f'interval_width = {iw}')

fig.suptitle('interval_width 对区间宽度的影响', fontsize=14)
plt.tight_layout()
plt.show()

# ============================================================
# 4. 历史区间覆盖率验证
# ============================================================
forecast_hist = forecast[forecast['ds'].isin(df['ds'])].copy()
forecast_hist = forecast_hist.merge(df, on='ds')

covered = (forecast_hist['y'] >= forecast_hist['yhat_lower']) & \
          (forecast_hist['y'] <= forecast_hist['yhat_upper'])
coverage = covered.mean()

print(f"\n历史数据区间覆盖率: {coverage:.1%} (期望 ~80%)")

# 按年份看覆盖情况
forecast_hist['year'] = forecast_hist['ds'].dt.year
for year, group in forecast_hist.groupby('year'):
    cov = ((group['y'] >= group['yhat_lower']) & (group['y'] <= group['yhat_upper'])).mean()
    print(f"  {year}: {cov:.1%}")

# 异常天数
outliers = forecast_hist[~covered]
print(f"\n超出预测区间的天数: {len(outliers)} / {len(forecast_hist)}")
if len(outliers) > 0:
    print("示例:")
    print(outliers[['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper']].head(10))

# ============================================================
# 5. MCMC 模式（可选，需要 cmdstanpy 或 pystan）
# ============================================================
try:
    print("\n尝试 MCMC 采样...")
    model_mcmc = Prophet(mcmc_samples=100, uncertainty_samples=200)
    model_mcmc.fit(df)
    future_mcmc = model_mcmc.make_future_dataframe(365)
    forecast_mcmc = model_mcmc.predict(future_mcmc)

    # 比较 MAP 和 MCMC 的预测区间宽度
    map_width = (forecast['yhat_upper'] - forecast['yhat_lower']).mean()
    mcmc_width = (forecast_mcmc['yhat_upper'] - forecast_mcmc['yhat_lower']).mean()

    print(f"  MAP  平均区间宽度: {map_width:.1f}")
    print(f"  MCMC 平均区间宽度: {mcmc_width:.1f}")
    print(f"  MCMC 比 MAP 宽 {(mcmc_width / map_width - 1) * 100:.0f}%（因为包含了参数不确定性）")
except Exception as e:
    print(f"  MCMC 不可用: {e}")
    print("  提示: 安装 cmdstanpy 后可使用 MCMC: uv pip install cmdstanpy")
    print("        然后使用: Prophet(stan_backend='CMDSTANPY', mcmc_samples=300)")

# ============================================================
# 6. "喇叭口"效应可视化
# ============================================================
horizon_days = (forecast['ds'] - df['ds'].max()).dt.days
forecast['horizon'] = horizon_days
forecast['interval_width'] = forecast['yhat_upper'] - forecast['yhat_lower']

fig, ax = plt.subplots(figsize=(12, 4))
ax.scatter(forecast['horizon'], forecast['interval_width'], alpha=0.3, s=5)
ax.set_xlabel('预测 Horizon (天)')
ax.set_ylabel('预测区间宽度')
ax.set_title('"喇叭口"效应 — 预测越远，区间越宽')
plt.tight_layout()
plt.show()
