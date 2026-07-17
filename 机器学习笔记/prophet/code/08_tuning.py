"""
第 9 章：超参数调优
====================
- 分阶段调参（mode → cps×sps → 精细）
- 热力图可视化
- 完整调优工作流
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import matplotlib.pyplot as plt
import itertools

# ============================================================
# 0. 数据准备
# ============================================================
url = "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_air_passengers.csv"
df = pd.read_csv(url)
df['ds'] = pd.to_datetime(df['ds'])
print(f"数据量: {len(df)}")

# ============================================================
# 1. 评估函数
# ============================================================
CV_CONFIG = {
    'initial': '1825 days',
    'period': '365 days',
    'horizon': '365 days',
}

def cv_rmse(params, df):
    """给定参数字典，返回交叉验证 RMSE"""
    model = Prophet(**params)
    model.fit(df)
    df_cv = cross_validation(model, **CV_CONFIG)
    df_metrics = performance_metrics(df_cv, rolling_window=0.1)
    return df_metrics['rmse'].mean()

# ============================================================
# 阶段 1: 加性 vs 乘性
# ============================================================
print("=" * 50)
print("阶段 1: 选择 seasonality_mode")
print("=" * 50)

results_mode = {}
for mode in ['additive', 'multiplicative']:
    rmse = cv_rmse({'seasonality_mode': mode}, df)
    results_mode[mode] = rmse
    print(f"  {mode:>15}: RMSE = {rmse:.2f}")

best_mode = min(results_mode, key=results_mode.get)
print(f"  → 选择 {best_mode}\n")

# ============================================================
# 阶段 2: changepoint_prior_scale × seasonality_prior_scale 联合调优
# ============================================================
print("=" * 50)
print("阶段 2: 趋势 × 季节性 联合搜索")
print("=" * 50)

cps_grid = [0.01, 0.05, 0.1, 0.5]
sps_grid = [1.0, 5.0, 10.0, 20.0]
stage2_results = []

for i, (cps, sps) in enumerate(itertools.product(cps_grid, sps_grid)):
    params = {
        'seasonality_mode': best_mode,
        'changepoint_prior_scale': cps,
        'seasonality_prior_scale': sps,
    }
    rmse = cv_rmse(params, df)
    stage2_results.append({'cps': cps, 'sps': sps, 'rmse': rmse})
    print(f"  [{i+1}/{len(cps_grid)*len(sps_grid)}] cps={cps:.2f}, sps={sps:.0f} → RMSE={rmse:.1f}")

df_s2 = pd.DataFrame(stage2_results)
best_s2 = df_s2.loc[df_s2['rmse'].idxmin()]
print(f"\n  → 最佳: cps={best_s2['cps']}, sps={best_s2['sps']:.0f}, RMSE={best_s2['rmse']:.1f}\n")

# ============================================================
# 阶段 2b: 热力图
# ============================================================
pivot = df_s2.pivot_table(values='rmse', index='cps', columns='sps')

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.pcolormesh(pivot.columns, pivot.index, pivot.values,
                   cmap='RdYlGn_r', shading='auto')
for i, cps in enumerate(pivot.index):
    for j, sps in enumerate(pivot.columns):
        ax.text(sps, cps, f'{pivot.values[i, j]:.0f}',
                ha='center', va='center', fontsize=11,
                fontweight='bold' if pivot.values[i, j] == pivot.values.min().min() else 'normal')
ax.set_xlabel('seasonality_prior_scale')
ax.set_ylabel('changepoint_prior_scale')
ax.set_title(f'阶段2: RMSE 热力图 (mode={best_mode})\n★ = 最优')
plt.colorbar(im, ax=ax, label='RMSE')
plt.tight_layout()
plt.show()

# ============================================================
# 阶段 3: 细调 — 节假日 + 傅里叶阶数
# ============================================================
print("=" * 50)
print("阶段 3: 精细调优")
print("=" * 50)

best_params_base = {
    'seasonality_mode': best_mode,
    'changepoint_prior_scale': best_s2['cps'],
    'seasonality_prior_scale': best_s2['sps'],
}

# 节假日先验
print("  holidays_prior_scale:")
for hps in [5.0, 10.0, 15.0]:
    params = {**best_params_base, 'holidays_prior_scale': hps}
    print(f"    {hps:.0f} → RMSE = {cv_rmse(params, df):.1f}")

# 年季节阶数
print("  yearly_seasonality (傅里叶阶数):")
for yf in [10, 15, 20]:
    params = {**best_params_base, 'yearly_seasonality': yf}
    print(f"    {yf:>2} → RMSE = {cv_rmse(params, df):.1f}")

print()

# ============================================================
# 阶段 4: 最终模型 vs 基准
# ============================================================
print("=" * 50)
print("阶段 4: 最终验证")
print("=" * 50)

final_model = Prophet(**best_params_base)
final_model.fit(df)
df_cv_final = cross_validation(final_model, **CV_CONFIG)
final_metrics = performance_metrics(df_cv_final, rolling_window=0.1)

print(f"  最终 RMSE:  {final_metrics['rmse'].mean():.1f}")
print(f"  最终 MAE:   {final_metrics['mae'].mean():.1f}")
print(f"  最终 MAPE:  {final_metrics['mape'].mean():.1f}%")
print(f"  基准 RMSE:  {results_mode[best_mode]:.1f}")
print(f"  提升:       {(1 - final_metrics['rmse'].mean() / results_mode[best_mode]) * 100:.1f}%")

print(f"\n  最优参数:")
for k, v in best_params_base.items():
    print(f"    {k} = {v}")
