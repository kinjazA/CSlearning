"""
第 3 章：趋势建模
==================
- 线性 vs 逻辑增长
- 变点检测与可视化
- changepoint_prior_scale 对比实验
- 手动指定变点
"""

import pandas as pd
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
import matplotlib.pyplot as plt

# ============================================================
# 0. 数据准备
# ============================================================
url = "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv"
df = pd.read_csv(url)
print(f"数据形状: {df.shape}, 日期范围: {df['ds'].min()} ~ {df['ds'].max()}")

# ============================================================
# 1. 基线模型 — 线性增长 + 默认变点
# ============================================================
model = Prophet(growth='linear')
model.fit(df)
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# 可视化变点
fig, ax = plt.subplots(figsize=(14, 5))
ax = model.plot(forecast, ax=ax)
add_changepoints_to_plot(ax, model, forecast)
ax.set_title('线性增长 + 自动变点检测')
plt.tight_layout()
plt.show()

# 查看检测到的变点
print("\n检测到的变点:")
changepoints = model.changepoints
deltas = model.params['delta'].mean(0)
for cp, d in zip(changepoints, deltas):
    if abs(d) > 1e-4:
        print(f"  {cp.date()}: δ = {d:.4f}")

# ============================================================
# 2. 逻辑增长实验
# ============================================================
df['cap'] = 10.5   # 人工设定饱和上限
df['floor'] = 0    # 下限

model_logistic = Prophet(growth='logistic')
model_logistic.fit(df)
future_logistic = model_logistic.make_future_dataframe(periods=365)
future_logistic['cap'] = 10.5
forecast_logistic = model_logistic.predict(future_logistic)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
model.plot(forecast, ax=axes[0])
axes[0].set_title('线性增长')
model_logistic.plot(forecast_logistic, ax=axes[1])
axes[1].set_title('逻辑增长 (cap=10.5)')
plt.tight_layout()
plt.show()

# ============================================================
# 3. changepoint_prior_scale 对比实验
# ============================================================
cps_values = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for ax, cps in zip(axes, cps_values):
    m = Prophet(growth='linear', changepoint_prior_scale=cps, yearly_seasonality=False)
    m.fit(df)
    fc = m.predict(m.make_future_dataframe(365))
    ax.plot(df['ds'], df['y'], '.', alpha=0.3, markersize=2, color='gray')
    ax.plot(fc['ds'], fc['trend'], 'r-', lw=2)
    ax.set_title(f'cps = {cps}')
    ax.set_ylabel('Trend')

fig.suptitle('changepoint_prior_scale 对趋势的影响', fontsize=14)
plt.tight_layout()
plt.show()

# ============================================================
# 4. 手动指定变点
# ============================================================
manual_changepoints = ['2011-01-01', '2013-01-01', '2015-01-01']
model_manual = Prophet(changepoints=manual_changepoints)
model_manual.fit(df)
future_manual = model_manual.make_future_dataframe(365)
forecast_manual = model_manual.predict(future_manual)

fig, ax = plt.subplots(figsize=(14, 5))
ax = model_manual.plot(forecast_manual, ax=ax)
add_changepoints_to_plot(ax, model_manual, forecast_manual)
for cp in manual_changepoints:
    ax.axvline(pd.to_datetime(cp), color='green', linestyle='--', alpha=0.5, label='手动变点' if cp == manual_changepoints[0] else '')
ax.legend()
ax.set_title('手动指定变点')
plt.tight_layout()
plt.show()

# ============================================================
# 5. 变点范围 (changepoint_range) 实验
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, cr in zip(axes, [0.5, 0.8, 0.95]):
    m = Prophet(changepoint_range=cr, yearly_seasonality=False)
    m.fit(df)
    fc = m.predict(m.make_future_dataframe(365))
    ax.plot(df['ds'], df['y'], '.', alpha=0.3, markersize=2, color='gray')
    ax.plot(fc['ds'], fc['trend'], 'r-')
    ax.axvline(df['ds'].iloc[int(len(df) * cr)], color='green', linestyle='--',
               label=f'检测范围边界 (前{cr*100:.0f}%)')
    ax.set_title(f'changepoint_range = {cr}')
    ax.legend()
plt.tight_layout()
plt.show()
