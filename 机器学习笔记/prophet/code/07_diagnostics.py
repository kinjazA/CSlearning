"""
第 8 章：模型诊断与评估
========================
- 时间序列交叉验证
- 性能指标计算（RMSE, MAE, MAPE 等）
- 残差分析与诊断
- 区间覆盖率检查
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import matplotlib.pyplot as plt

# ============================================================
# 0. 数据准备
# ============================================================
url = "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_air_passengers.csv"
df = pd.read_csv(url)
df['ds'] = pd.to_datetime(df['ds'])
print(f"数据量: {len(df)}, 日期范围: {df['ds'].min().date()} ~ {df['ds'].max().date()}")

# ============================================================
# 1. 建模
# ============================================================
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    changepoint_prior_scale=0.05,
)
model.fit(df)

# ============================================================
# 2. 交叉验证
# ============================================================
# 注: 航空乘客数据只有 144 个月，这里用较小的参数做演示
df_cv = cross_validation(
    model,
    initial='1825 days',   # 初始训练 ≥ 5 年
    period='365 days',     # 每年一个 cutoff
    horizon='365 days',    # 预测 1 年
)

print(f"\n交叉验证结果:")
print(f"  Cutoff 数: {df_cv['cutoff'].nunique()}")
print(f"  总预测记录: {len(df_cv)}")
print(f"  df_cv 列: {df_cv.columns.tolist()}")
print(f"\n  示例数据:")
print(df_cv[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'y', 'cutoff']].head(10))

# ============================================================
# 3. 性能指标
# ============================================================
df_metrics = performance_metrics(df_cv, rolling_window=0.1)
print(f"\n性能指标 (按 horizon):")
print(df_metrics[['horizon', 'mse', 'rmse', 'mae', 'mape', 'mdape']].to_string())

# 汇总指标
print(f"\n汇总:")
print(f"  平均 RMSE:  {df_metrics['rmse'].mean():.1f}")
print(f"  平均 MAE:   {df_metrics['mae'].mean():.1f}")
print(f"  平均 MAPE:  {df_metrics['mape'].mean():.1f}%")
print(f"  平均 MDAPE: {df_metrics['mdape'].mean():.1f}%")

# ============================================================
# 4. 按 horizon 可视化精度衰减
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, metric in zip(axes.flat, ['rmse', 'mae', 'mape', 'mdape']):
    ax.plot(df_metrics['horizon'], df_metrics[metric], 'o-', markersize=4)
    ax.set_title(f'{metric.upper()} vs Horizon')
    ax.set_xlabel('Horizon (天)')
    ax.set_ylabel(metric.upper())
    ax.grid(True, alpha=0.3)

plt.suptitle('预测精度随 Horizon 的衰减', fontsize=14)
plt.tight_layout()
plt.show()

# ============================================================
# 5. 残差分析
# ============================================================
df_cv['residual'] = df_cv['y'] - df_cv['yhat']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5a. 残差时序图
ax = axes[0][0]
ax.scatter(df_cv['ds'], df_cv['residual'], alpha=0.3, s=10)
ax.axhline(y=0, color='r', linestyle='--')
ax.set_title('残差时序图')
ax.set_xlabel('Date')
ax.set_ylabel('Residual')

# 5b. 残差分布
ax = axes[0][1]
ax.hist(df_cv['residual'], bins=40, edgecolor='white')
ax.axvline(x=0, color='r', linestyle='--')
ax.set_title(f'残差分布 (均值={df_cv["residual"].mean():.1f}, 标准差={df_cv["residual"].std():.1f})')

# 5c. 残差 vs 预测值
ax = axes[1][0]
ax.scatter(df_cv['yhat'], df_cv['residual'], alpha=0.3, s=10)
ax.axhline(y=0, color='r', linestyle='--')
ax.set_title('残差 vs 预测值（异方差检查）')
ax.set_xlabel('Predicted (yhat)')
ax.set_ylabel('Residual')

# 5d. 按 cutoff 的误差
ax = axes[1][1]
df_cv.groupby('cutoff')['residual'].apply(
    lambda x: np.sqrt((x ** 2).mean())
).plot(kind='bar', ax=ax)
ax.set_title('每个 Cutoff 的 RMSE')
ax.set_xlabel('Cutoff')
ax.set_ylabel('RMSE')
ax.tick_params(axis='x', rotation=45)

plt.suptitle('残差诊断', fontsize=14)
plt.tight_layout()
plt.show()

# ============================================================
# 6. 区间覆盖率检查
# ============================================================
covered = (df_cv['y'] >= df_cv['yhat_lower']) & (df_cv['y'] <= df_cv['yhat_upper'])
coverage = covered.mean()

print(f"\n区间覆盖率: {coverage:.1%} (期望 ~80%)")

if coverage < 0.70:
    print("  ⚠️ 覆盖率偏低 — 模型过于自信，区间偏窄")
elif coverage > 0.90:
    print("  ⚠️ 覆盖率偏高 — 模型过于保守，区间偏宽")
else:
    print("  ✅ 覆盖率合理")

# 按 cutoff 看覆盖
print("\n各 cutoff 的覆盖率:")
for cutoff, group in df_cv.groupby('cutoff'):
    cov = ((group['y'] >= group['yhat_lower']) & (group['y'] <= group['yhat_upper'])).mean()
    print(f"  {cutoff.date()}: {cov:.1%}")

# ============================================================
# 7. 残差自相关检查
# ============================================================
from pandas.plotting import autocorrelation_plot

fig, ax = plt.subplots(figsize=(12, 4))
autocorrelation_plot(df_cv['residual'].dropna(), ax=ax)
ax.set_title('残差自相关 — 理想情况下应在置信区间内')
ax.set_ylim(-0.2, 0.5)
plt.tight_layout()
plt.show()

# ============================================================
# 8. 诊断总结
# ============================================================
print("\n" + "=" * 50)
print("诊断总结")
print("=" * 50)
residual_mean = df_cv['residual'].mean()
residual_std = df_cv['residual'].std()
y_std = df['y'].std()

print(f"  残差均值: {residual_mean:.1f} {'⚠️ 有系统偏差' if abs(residual_mean) > y_std * 0.1 else '✅'}")
print(f"  区间覆盖率: {coverage:.1%} {'✅' if 0.65 < coverage < 0.90 else '⚠️'}")
print(f"  残差标准差: {residual_std:.1f} (y 标准差 = {y_std:.1f})")
print(f"  变异系数(CV): {residual_std / y_std:.2f}")
