"""
第 4 章：季节性建模
====================
- 傅里叶阶数对比
- 自定义季节性
- 加性 vs 乘性季节性
- 条件季节性
"""

import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# ============================================================
# 0. 数据准备
# ============================================================
url = "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_air_passengers.csv"
df = pd.read_csv(url)
print(f"数据形状: {df.shape}")

# ============================================================
# 1. 傅里叶阶数对比 — 年季节性
# ============================================================
fourier_orders = [3, 5, 10, 20]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, N in zip(axes, fourier_orders):
    m = Prophet(yearly_seasonality=N, weekly_seasonality=False)
    m.fit(df)
    fc = m.predict(m.make_future_dataframe(365))
    ax.plot(fc['ds'], fc['yearly'], lw=1.5)
    ax.set_title(f'年季节性 — 傅里叶阶数 N={N}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Yearly effect')

fig.suptitle('傅里叶阶数对年季节性形状的影响', fontsize=14)
plt.tight_layout()
plt.show()

# ============================================================
# 2. 自定义季节性 — 月度周期
# ============================================================
model = Prophet(
    yearly_seasonality=10,
    weekly_seasonality=3,
)

# 自定义月度季节性
model.add_seasonality(
    name='monthly',
    period=30.5,
    fourier_order=5,
)

model.fit(df)
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# 查看自定义季节性的贡献
fig = model.plot_components(forecast)
plt.suptitle('成分分解（含自定义月度季节性）')
plt.tight_layout()
plt.show()

print("季节性相关列:")
seasonal_cols = [c for c in forecast.columns if c not in ('ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend')]
print(seasonal_cols)
print(forecast[['ds'] + [c for c in seasonal_cols if c in forecast.columns]].tail(5))

# ============================================================
# 3. 加性 vs 乘性季节性 — 直观对比
# ============================================================
# 构造一个趋势增长明显的数据（乘性效应更容易显现）
import numpy as np
np.random.seed(42)

dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
n = len(dates)
trend = np.linspace(100, 10000, n)  # 趋势从 100 涨到 10000（100倍）
weekly = 0.3 * np.sin(2 * np.pi * np.arange(n) / 7)
noise = np.random.randn(n) * 50

df_synthetic = pd.DataFrame({
    'ds': dates,
    'y_additive': trend + weekly * 200 + noise,
    'y_multiplicative': trend * (1 + weekly) + noise,
})

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

for idx, (mode, y_col, title) in enumerate([
    ('additive', 'y_additive', '加性数据（波动固定）'),
    ('multiplicative', 'y_multiplicative', '乘性数据（波动随趋势放大）'),
]):
    # 原始数据
    ax0 = axes[0][idx]
    ax0.plot(dates[:365], df_synthetic[y_col].iloc[:365])
    ax0.set_title(f'原始数据 — {title}')

    # 用加性模型预测
    ax1 = axes[1][idx]
    m = Prophet(seasonality_mode=mode, yearly_seasonality=False, weekly_seasonality=5)
    m.fit(df_synthetic.rename(columns={y_col: 'y'}))
    fc = m.predict(m.make_future_dataframe(90))
    ax1.plot(fc['ds'], fc['yhat'], 'b-', label='Prediction')
    ax1.plot(fc['ds'], fc['yhat_lower'], 'b--', alpha=0.3)
    ax1.plot(fc['ds'], fc['yhat_upper'], 'b--', alpha=0.3)
    ax1.set_title(f'{mode} 模型预测')

plt.tight_layout()
plt.show()

# ============================================================
# 4. 季节性先验尺度对比
# ============================================================
sps_values = [0.01, 1.0, 10.0, 50.0]

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, sps in zip(axes, sps_values):
    m = Prophet(seasonality_prior_scale=sps, weekly_seasonality=False)
    m.fit(df)
    fc = m.predict(m.make_future_dataframe(365))
    ax.plot(fc['ds'], fc['yearly'], lw=1.5)
    ax.set_title(f'seasonality_prior_scale = {sps}')
    ax.set_ylabel('Yearly effect')

fig.suptitle('seasonality_prior_scale 对年季节性幅度的影响', fontsize=14)
plt.tight_layout()
plt.show()
