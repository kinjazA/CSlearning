"""
第 5 章：节假日与特殊事件
==========================
- 内置国家节假日
- 自定义节假日（含窗口）
- 节假日效应可视化
- 节假日先验尺度
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

# ============================================================
# 0. 生成模拟数据 — 带促销效应的日销售
# ============================================================
np.random.seed(42)
dates = pd.date_range('2021-01-01', '2024-12-31', freq='D')
n = len(dates)

# 基线趋势 + 季节
trend = np.linspace(2000, 3500, n)
weekly = 300 * np.sin(2 * np.pi * np.arange(n) / 7)
yearly = 400 * np.sin(2 * np.pi * np.arange(n) / 365.25)
noise = np.random.randn(n) * 80

y = trend + weekly + yearly + noise

# 模拟促销效应
promo_dates = []
for year in [2021, 2022, 2023]:
    # 618 大促
    promo_dates.extend(pd.date_range(f'{year}-06-15', f'{year}-06-20', freq='D'))
    # 双11 大促
    promo_dates.extend(pd.date_range(f'{year}-11-09', f'{year}-11-13', freq='D'))
    # 春节前
    spring_festival = {
        2021: '2021-01-20', 2022: '2022-01-25', 2023: '2023-01-15'
    }[year]
    promo_dates.extend(pd.date_range(spring_festival, periods=10, freq='D'))

for d in promo_dates:
    if d in dates:
        idx = dates.get_loc(d)
        y[idx] += 1500  # 促销日涨 1500

df = pd.DataFrame({'ds': dates, 'y': y})
print(f"模拟数据: {len(df)} 天, y 范围: [{y.min():.0f}, {y.max():.0f}]")

# ============================================================
# 1. 不加节假日 — 查看残差
# ============================================================
model_no_holiday = Prophet(yearly_seasonality=15, weekly_seasonality=5)
model_no_holiday.fit(df)
forecast_no_h = model_no_holiday.predict(df[['ds']])
residuals_no_h = df['y'] - forecast_no_h['yhat']

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
axes[0].scatter(df['ds'], residuals_no_h, alpha=0.3, s=10)
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_title('不加节假日 — 残差（促销日有明显的正残差）')
axes[0].set_ylabel('Residual')
plt.tight_layout()
plt.show()

# ============================================================
# 2. 加入自定义节假日 — 含窗口
# ============================================================
promo_events = pd.DataFrame({
    'holiday': [
        '618促销', '618促销', '618促销',
        '双11促销', '双11促销', '双11促销',
        '春节前采购', '春节前采购', '春节前采购',
    ],
    'ds': pd.to_datetime([
        '2021-06-18', '2022-06-18', '2023-06-18',
        '2021-11-11', '2022-11-11', '2023-11-11',
        '2021-01-20', '2022-01-25', '2023-01-15',
    ]),
    'lower_window': [-3, -3, -3, -2, -2, -2, -10, -10, -10],
    'upper_window': [2, 2, 2, 2, 2, 2, 2, 2, 2],
})

model = Prophet(
    holidays=promo_events,
    holidays_prior_scale=15.0,
    yearly_seasonality=15,
    weekly_seasonality=5,
)
model.fit(df)

# 查看节假日表
print(f"\n模型识别的节假日: {model.train_holiday_names}")

future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

# ============================================================
# 3. 节假日效应可视化
# ============================================================
fig = model.plot_components(forecast)
plt.suptitle('成分分解 — 含自定义节假日')
plt.tight_layout()
plt.show()

# 查看节假日效应列
holiday_cols = [c for c in forecast.columns if c in model.train_holiday_names or c == 'holidays']
if 'holidays' in forecast.columns:
    holiday_days = forecast[forecast['holidays'] != 0][['ds', 'holidays']]
    print(f"\n节假日的总效应 (holidays 列):")
    print(holiday_days.head(20))

# ============================================================
# 4. 添加内置中国节假日（叠加）
# ============================================================
model_cn = Prophet(
    holidays=promo_events,
    holidays_prior_scale=15.0,
    yearly_seasonality=15,
    weekly_seasonality=5,
)
model_cn.add_country_holidays(country_name='CN')
model_cn.fit(df)
forecast_cn = model_cn.predict(future)

print(f"\n叠加内置中国节假日后: {model_cn.train_holiday_names}")

# ============================================================
# 5. holidays_prior_scale 对比
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
for ax, hps in zip(axes, [1.0, 10.0, 50.0]):
    m = Prophet(holidays=promo_events, holidays_prior_scale=hps,
                yearly_seasonality=15, weekly_seasonality=5)
    m.fit(df)
    fc = m.predict(future)
    holiday_effect = fc[[c for c in fc.columns if c in m.train_holiday_names]].sum(axis=1)
    ax.plot(fc['ds'], holiday_effect)
    ax.set_title(f'holidays_prior_scale = {hps}')
    ax.set_ylabel('Total Holiday Effect')

fig.suptitle('holidays_prior_scale 对节假日效应大小的影响', fontsize=14)
plt.tight_layout()
plt.show()
