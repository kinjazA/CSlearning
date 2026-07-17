"""
第 6 章：额外回归量
====================
- 基本用法：注册回归量 + 填充未来值
- 加性 vs 乘性回归量
- 非线性关系的特征工程
- 完整示例：餐厅日营收预测
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

# ============================================================
# 0. 生成模拟数据 — 餐厅日营收
# ============================================================
np.random.seed(42)
dates = pd.date_range('2022-01-01', '2024-12-31', freq='D')
n = len(dates)

# 基线
trend = np.linspace(8000, 12000, n)
weekly = 2000 * np.sin(2 * np.pi * np.arange(n) / 7)
yearly = 1500 * np.sin(2 * np.pi * np.arange(n) / 365.25)

# 外部变量
promo = np.where(np.arange(n) % 30 < 3, 1, 0)                         # 每月前3天促销
ad_spend = np.random.gamma(2, 200, n)                                 # 广告投放金额
temperature = 15 + 10 * np.sin(2 * np.pi * np.arange(n) / 365.25) \
              + np.random.randn(n) * 2                                # 日平均温度

# 真实关系
promo_effect = promo * 3000                                           # 促销 +3000
ad_effect = ad_spend * 0.5                                            # 每元广告 → 0.5 营收
# U 形温度效应：偏离 25°C 越远，负面影响越大
temp_effect = -np.abs(temperature - 25) * 100
noise = np.random.randn(n) * 500

y = trend + weekly + yearly + promo_effect + ad_effect + temp_effect + noise
y = np.maximum(y, 100)  # 营收不能为负

df = pd.DataFrame({
    'ds': dates,
    'y': y,
    'promo': promo,
    'ad_spend': ad_spend,
    'temperature': temperature,
})
# 非线性特征
df['temp_deviation'] = np.abs(df['temperature'] - 25)

print(f"数据量: {len(df)} 天")
print(f"y 范围: {y.min():.0f} ~ {y.max():.0f}")

# 划分训练/测试（保留最后 90 天）
train = df.iloc[:-90].copy()
test  = df.iloc[-90:].copy()

# ============================================================
# 1. 基线模型 — 无额外回归量
# ============================================================
model_baseline = Prophet(yearly_seasonality=15, weekly_seasonality=5)
model_baseline.fit(train)
future_baseline = model_baseline.make_future_dataframe(90)
forecast_baseline = model_baseline.predict(future_baseline)

# ============================================================
# 2. 加入额外回归量
# ============================================================
model = Prophet(
    yearly_seasonality=15,
    weekly_seasonality=5,
    changepoint_prior_scale=0.05,
)

# 注册回归量 — 必须在 fit 前调用
model.add_regressor('promo')
model.add_regressor('ad_spend')
model.add_regressor('temperature')
model.add_regressor('temp_deviation')

model.fit(train)

# ============================================================
# 3. ⚠️ 关键步骤：填充未来值
# ============================================================
future = model.make_future_dataframe(periods=90)

# 从测试集填充（实际场景中需要单独提供未来值）
future = future.merge(
    test[['ds', 'promo', 'ad_spend', 'temperature', 'temp_deviation']],
    on='ds', how='left'
)

# 填充可能的缺失值（如果 future 包含不在 test 中的日期）
future['promo'].fillna(0, inplace=True)
future['ad_spend'].fillna(future['ad_spend'].mean(), inplace=True)
future['temperature'].fillna(future['temperature'].mean(), inplace=True)
future['temp_deviation'].fillna(future['temp_deviation'].mean(), inplace=True)

forecast = model.predict(future)

# ============================================================
# 4. 评估：基线 vs 带回归量
# ============================================================
comparison_baseline = test[['ds', 'y']].merge(
    forecast_baseline[['ds', 'yhat']], on='ds'
)
comparison_model = test[['ds', 'y']].merge(
    forecast[['ds', 'yhat']], on='ds'
)

mae_baseline = (comparison_baseline['y'] - comparison_baseline['yhat']).abs().mean()
mae_model    = (comparison_model['y'] - comparison_model['yhat']).abs().mean()

print(f"\n基线模型 (无回归量) MAE: {mae_baseline:.0f}")
print(f"完整模型 (含回归量) MAE: {mae_model:.0f}")
print(f"MAE 提升: {(1 - mae_model / mae_baseline) * 100:.1f}%")

# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5a. 预测 vs 实际
ax = axes[0][0]
ax.plot(test['ds'], test['y'], 'k-', label='Actual', alpha=0.7)
ax.plot(comparison_baseline['ds'], comparison_baseline['yhat'], 'r--', label='Baseline', alpha=0.7)
ax.plot(comparison_model['ds'], comparison_model['yhat'], 'b-', label='With Regressors', alpha=0.7)
ax.legend()
ax.set_title('预测对比 — 测试集 90 天')

# 5b. 残差对比
ax = axes[0][1]
ax.hist(comparison_baseline['y'] - comparison_baseline['yhat'],
        bins=30, alpha=0.5, label='Baseline errors', color='red')
ax.hist(comparison_model['y'] - comparison_model['yhat'],
        bins=30, alpha=0.5, label='Model errors', color='blue')
ax.axvline(0, color='black', linestyle='--')
ax.legend()
ax.set_title('残差分布对比')

# 5c. 各回归量的贡献
ax = axes[1][0]
regressor_cols = ['promo', 'ad_spend', 'temperature', 'temp_deviation']
(forecast[['ds'] + regressor_cols]
 .set_index('ds')
 .plot(ax=ax))
ax.set_title('各回归量对预测的贡献')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 5d. 温度 U 形效应验证
ax = axes[1][1]
ax.scatter(df['temperature'], df['temp_deviation'] * 100, alpha=0.3, s=5)
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Deviation from 25°C')
ax.set_title('温度 U 形效应的特征工程')

plt.tight_layout()
plt.show()

# ============================================================
# 6. 查看回归量的系数（近似）
# ============================================================
print("\n回归量系数（训练后的均值估计）:")
for reg in ['promo', 'ad_spend', 'temperature', 'temp_deviation']:
    beta_col = f'beta_{reg}' if f'beta_{reg}' in model.params else None
    if reg in forecast.columns:
        mean_effect = forecast[reg].mean()
        print(f"  {reg}: 平均效应 = {mean_effect:.1f}")
