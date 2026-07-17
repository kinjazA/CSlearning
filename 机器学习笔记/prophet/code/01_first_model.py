"""
第 2 章：Prophet 快速入门 — 第一个模型
==========================================
跑通 fit → predict → plot 全流程，理解预测输出的每一列。
"""

import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# ============================================================
# 1. 准备数据 — 经典航空乘客数据
# ============================================================
url = "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_air_passengers.csv"
df = pd.read_csv(url)
print(f"数据形状: {df.shape}")
print(f"日期范围: {df['ds'].min()} ~ {df['ds'].max()}")
print(df.head())

# ============================================================
# 2. 拟合模型
# ============================================================
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
)
model.fit(df)
print("\n模型拟合完成！")

# ============================================================
# 3. 生成未来预测
# ============================================================
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# 只看未来部分的关键列
future_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10)
print("\n未来 10 天预测:")
print(future_forecast.to_string())

# ============================================================
# 4. 查看预测输出的全部列
# ============================================================
print("\nforecast 包含的列:")
print(forecast.columns.tolist())

# 查看某一天的完整分解
sample = forecast[forecast['ds'] == '1961-01-01']
print("\n1961-01-01 的完整预测分解:")
for col in ['yhat', 'trend', 'yearly', 'weekly', 'additive_terms']:
    print(f"  {col}: {sample[col].values[0]:.1f}")

# ============================================================
# 5. 可视化
# ============================================================
fig1 = model.plot(forecast)
plt.title('Prophet Forecast — Airline Passengers')
plt.xlabel('Date')
plt.ylabel('Passengers')
plt.tight_layout()
plt.show()

fig2 = model.plot_components(forecast)
plt.tight_layout()
plt.show()
