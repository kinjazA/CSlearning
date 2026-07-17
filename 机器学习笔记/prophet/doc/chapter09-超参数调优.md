# 第 9 章：超参数调优

前面八章你学会了建模和诊断。这一章回答：**给定我的数据，哪组参数最好？**

调参不是玄学，也**不能只靠"感觉"**——需要一套系统的方法论，用第 8 章的交叉验证结果来驱动决策。

---

## 9.1 参数全景图

在开始调参之前，搞清楚哪些参数值得调、哪些通常不用动。

### 9.1.1 四个优先级

```
优先级 1 — 必须调的:
  ├── changepoint_prior_scale    趋势灵活性
  ├── seasonality_prior_scale    季节性灵活性
  └── seasonality_mode           加性 vs 乘性

优先级 2 — 经常调的:
  ├── holidays_prior_scale       节假日灵活性
  ├── yearly_seasonality         年季节傅里叶阶数
  ├── weekly_seasonality         周季节傅里叶阶数
  └── changepoint_range          变点检测范围

优先级 3 — 偶尔调的:
  ├── n_changepoints             候选变点数量
  ├── interval_width             预测区间宽度
  └── 额外回归量的 prior_scale

优先级 4 — 通常不动的:
  ├── growth                     线性 vs 逻辑（业务决定，不是调出来的）
  ├── uncertainty_samples        精度换时间，不是"调优"
  └── mcmc_samples               同上
```

### 9.1.2 参数之间的"势力范围"

调参时最重要的一个认知：**参数不是独立的**。它们之间会互相"抢地盘"。

```
changepoint_prior_scale 大
  → 趋势抢走了季节性本应解释的波动
  → 季节性分量被"压扁"
  → 需要同时增大 seasonality_prior_scale 来"对抗"

seasonality_prior_scale 大
  → 季节性过度灵活
  → 短期的噪声被当成季节模式
  → 趋势变得过于平滑（好波动都被季节抢走了）
```

> **调参第一原则**：趋势和季节性的先验尺度要**协同调整**，不能单独拧一个旋钮。

---

## 9.2 调参策略

### 9.2.1 分阶段调参法（推荐）

不要一次性在所有参数上做网格搜索——组合爆炸，而且很多组合没有意义。

```
阶段 1: 粗调 — 确定大方向（3-4 组参数）
  ├── 加性 + 默认参数 → 基准 RMSE
  ├── 乘性 + 默认参数 → 对比 RMSE
  └── 选胜出的 mode

阶段 2: 中调 — 趋势和季节性（5-10 组参数）
  ├── changepoint_prior_scale: [0.01, 0.05, 0.1, 0.5, 1.0]
  ├── seasonality_prior_scale: [1.0, 5.0, 10.0, 15.0, 20.0]
  └── 选胜出的组合

阶段 3: 细调 — 具体组件（3-5 组参数）
  ├── holidays_prior_scale: [5.0, 10.0, 15.0]
  ├── 傅里叶阶数微调
  └── 选最终参数

阶段 4: 验证 — 确保没有过拟合
  ├── 在最新的 cutoff 上验证精度和残差
  └── 与基准模型对比
```

### 9.2.2 为什么不能一上来就网格搜索

```python
# ❌ 反模式：一次性调全部
param_grid = {
    'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
    'seasonality_prior_scale': [0.1, 1.0, 5.0, 10.0, 20.0, 50.0],
    'holidays_prior_scale': [0.1, 1.0, 5.0, 10.0, 20.0],
    'seasonality_mode': ['additive', 'multiplicative'],
    'yearly_seasonality': [5, 10, 15, 20],
    'weekly_seasonality': [3, 5, 7],
}
# 组合数: 7 × 6 × 5 × 2 × 4 × 3 = 5,040 种
# 每种都需要做 cv → 完全不现实
```

---

## 9.3 代码：手动调参框架

```python
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.model_selection import ParameterGrid
import itertools

def evaluate_params(df, param_dict, cv_kwargs):
    """
    给定一组参数，返回 CV 的 RMSE。
    """
    # 1. 创建模型
    model = Prophet(**param_dict)

    # 2. 如果需要乘性，注册节假日为乘性
    # （add_regressor 同理）

    # 3. 拟合
    model.fit(df)

    # 4. 交叉验证
    df_cv = cross_validation(model, **cv_kwargs)

    # 5. 计算指标
    df_metrics = performance_metrics(df_cv, rolling_window=0.1)

    # 6. 返回平均 RMSE（取所有 horizon 的均值）
    return df_metrics['rmse'].mean()


# ============================================================
# 阶段 1: 选 mode
# ============================================================
cv_config = {
    'initial': '1095 days',
    'period': '180 days',
    'horizon': '365 days',
}

results_mode = {}
for mode in ['additive', 'multiplicative']:
    rmse = evaluate_params(
        df,
        {'seasonality_mode': mode, 'yearly_seasonality': True, 'weekly_seasonality': True},
        cv_config,
    )
    results_mode[mode] = rmse
    print(f"mode={mode:>15}: RMSE={rmse:.2f}")

best_mode = min(results_mode, key=results_mode.get)
print(f"\n✅ 最佳 mode: {best_mode}")
```

---

## 9.4 阶段 2：趋势 × 季节性联合调优

```python
# ============================================================
# 阶段 2: 联合调优 changepoint_prior_scale × seasonality_prior_scale
# ============================================================

# 定义搜索空间
cps_values = [0.01, 0.05, 0.1, 0.5, 1.0]
sps_values = [1.0, 5.0, 10.0, 20.0]

results = []
total = len(cps_values) * len(sps_values)

for i, (cps, sps) in enumerate(itertools.product(cps_values, sps_values)):
    print(f"[{i+1}/{total}] cps={cps:.2f}, sps={sps:.1f} ...", end=' ')

    params = {
        'seasonality_mode': best_mode,
        'changepoint_prior_scale': cps,
        'seasonality_prior_scale': sps,
    }

    rmse = evaluate_params(df, params, cv_config)
    results.append({'cps': cps, 'sps': sps, 'rmse': rmse})
    print(f"RMSE={rmse:.2f}")

# 将结果转为透视表
results_df = pd.DataFrame(results)
pivot = results_df.pivot_table(
    values='rmse', index='cps', columns='sps'
)

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn_r', ax=ax)
ax.set_title(f'RMSE 热力图 — changepoint_prior_scale × seasonality_prior_scale\n(mode={best_mode})')
ax.set_xlabel('seasonality_prior_scale')
ax.set_ylabel('changepoint_prior_scale')
plt.tight_layout()
plt.show()

# 找最优组合
best_row = results_df.loc[results_df['rmse'].idxmin()]
print(f"\n✅ 最佳组合: cps={best_row['cps']}, sps={best_row['sps']}, RMSE={best_row['rmse']:.2f}")
```

---

## 9.5 热力图解读指南

调参热力图通常呈现三种形态，每种对应不同的数据特征：

```
形态 A: 单一低谷                    形态 B: 平坦区域
  sps→                              sps→
cps  1   5  10  20                cps  1   5  10  20
↓ 0.01 [120 118 116 115]          ↓ 0.01 [105 104 104 103]
  0.05 [110 108 106 105]            0.05 [104 103 102 101]
  0.1  [102 100  98  99]            0.1  [103 101  99  99] ← 平坦
  0.5  [105 106 110 115]            0.5  [104 102 100 101]
  1.0  [115 120 130 140]            1.0  [120 118 116 115]

  → 明确的最优区域                   → 多个组合效果接近
  → 数据季节+趋势模式清晰            → 数据噪声大，精细调节无意义
  → 选最优的就行                    → 选参数较简单的（偏小的 cps/spc）

形态 C: 对角线分布
  sps→
cps  1   5  10  20
↓ 0.01 [110 115 120 130]  ← 低 cps 高 sps = 差
  0.05 [105 108 112 118]
  0.1  [102 104 106 110]
  0.5  [108 104 100  98]  ← 高 cps 低 sps = 差
  1.0  [120 110 105 100]

  → 趋势和季节性在"抢地盘"
  → 需要平衡：cps 和 sps 同方向调整
```

---

## 9.6 阶段 3：精细化调节

在最优的 `cps` × `sps` 组合基础上，微调剩余参数：

```python
# ============================================================
# 阶段 3: 精细调优 — 节假日 + 傅里叶阶数
# ============================================================

# 3a. 节假日先验
best_params = {'cps': best_row['cps'], 'sps': best_row['sps']}
for hps in [5.0, 10.0, 15.0, 20.0]:
    params = {
        **best_params,
        'holidays_prior_scale': hps,
        'seasonality_mode': best_mode,
    }
    rmse = evaluate_params(df, params, cv_config)
    print(f"holidays_prior_scale={hps:.0f}: RMSE={rmse:.2f}")

# 3b. 傅里叶阶数
for yf in [10, 15, 20, 25]:
    params = {
        **best_params,
        'yearly_seasonality': yf,
        'seasonality_mode': best_mode,
    }
    rmse = evaluate_params(df, params, cv_config)
    print(f"yearly_seasonality={yf}: RMSE={rmse:.2f}")

for wf in [3, 5, 7, 10]:
    params = {
        **best_params,
        'weekly_seasonality': wf,
        'seasonality_mode': best_mode,
    }
    rmse = evaluate_params(df, params, cv_config)
    print(f"weekly_seasonality={wf}: RMSE={rmse:.2f}")
```

---

## 9.7 完整调优工作流

```python
# ============================================================
# Prophet 超参数调优 — 完整流程
# ============================================================

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------
# 0. 准备
# ----------------------------------------------------------
df = pd.read_csv('your_data.csv')
df['ds'] = pd.to_datetime(df['ds'])

CV_CONFIG = {
    'initial': '1095 days',
    'period': '180 days',
    'horizon': '365 days',
}

def cv_rmse(model, df):
    """快速评估函数"""
    df_cv = cross_validation(model, **CV_CONFIG)
    df_m = performance_metrics(df_cv, rolling_window=0.1)
    return df_m['rmse'].mean()

# ----------------------------------------------------------
# 阶段 1: 加性 vs 乘性
# ----------------------------------------------------------
print("=" * 50)
print("阶段 1: 选择 seasonality_mode")
print("=" * 50)

baseline = {}
for mode in ['additive', 'multiplicative']:
    m = Prophet(seasonality_mode=mode)
    m.fit(df)
    baseline[mode] = cv_rmse(m, df)
    print(f"  {mode}: RMSE = {baseline[mode]:.1f}")

best_mode = min(baseline, key=baseline.get)
print(f"  → 选择 {best_mode}\n")

# ----------------------------------------------------------
# 阶段 2: cps × sps 联合搜索
# ----------------------------------------------------------
print("=" * 50)
print("阶段 2: 趋势 × 季节性")
print("=" * 50)

cps_grid = [0.01, 0.05, 0.1, 0.5]
sps_grid = [1.0, 5.0, 10.0, 20.0]
stage2_results = []

for i, (cps, sps) in enumerate(itertools.product(cps_grid, sps_grid)):
    m = Prophet(
        seasonality_mode=best_mode,
        changepoint_prior_scale=cps,
        seasonality_prior_scale=sps,
    )
    m.fit(df)
    rmse = cv_rmse(m, df)
    stage2_results.append({'cps': cps, 'sps': sps, 'rmse': rmse})
    print(f"  [{i+1}/{len(cps_grid)*len(sps_grid)}] cps={cps:.2f}, sps={sps:.0f} → RMSE={rmse:.1f}")

df_s2 = pd.DataFrame(stage2_results)
best_s2 = df_s2.loc[df_s2['rmse'].idxmin()]
print(f"  → 最佳: cps={best_s2['cps']}, sps={best_s2['sps']:.0f}, RMSE={best_s2['rmse']:.1f}\n")

# 热力图
pivot = df_s2.pivot_table(values='rmse', index='cps', columns='sps')
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn_r', ax=ax)
ax.set_title(f'阶段2: RMSE热力图 (mode={best_mode})')
plt.tight_layout()
plt.savefig('tuning_heatmap.png', dpi=150)

# ----------------------------------------------------------
# 阶段 3: 细调（节假日 + 阶数）
# ----------------------------------------------------------
print("=" * 50)
print("阶段 3: 细调")
print("=" * 50)

best_params = {
    'seasonality_mode': best_mode,
    'changepoint_prior_scale': best_s2['cps'],
    'seasonality_prior_scale': best_s2['sps'],
}

# 节假日
print("  holidays_prior_scale:")
for hps in [5.0, 10.0, 15.0]:
    m = Prophet(**best_params, holidays_prior_scale=hps)
    m.fit(df)
    print(f"    {hps:.0f} → RMSE={cv_rmse(m, df):.1f}")

# ----------------------------------------------------------
# 阶段 4: 最终验证
# ----------------------------------------------------------
print("\n" + "=" * 50)
print("阶段 4: 最终模型验证")
print("=" * 50)

final_model = Prophet(
    seasonality_mode=best_mode,
    changepoint_prior_scale=best_s2['cps'],
    seasonality_prior_scale=best_s2['sps'],
    holidays_prior_scale=10.0,  # 选最优的
)

final_model.fit(df)
df_cv_final = cross_validation(final_model, **CV_CONFIG)
final_metrics = performance_metrics(df_cv_final, rolling_window=0.1)

print(f"  最终 RMSE: {final_metrics['rmse'].mean():.1f}")
print(f"  最终 MAE:  {final_metrics['mae'].mean():.1f}")
print(f"  最终 MAPE: {final_metrics['mape'].mean():.1f}%")
print(f"  基准 RMSE: {baseline[best_mode]:.1f}")
print(f"  提升:     {(1 - final_metrics['rmse'].mean() / baseline[best_mode]) * 100:.1f}%")
```

---

## 9.8 常见误区

| 误区 | 为什么错 | 正确做法 |
|------|---------|----------|
| 在全部历史数据上调参，不保留测试集 | 调参过程本身就会"泄漏"测试信息 | 用较早的 cutoff 调参，在最新的 cutoff 上验证 |
| 只看 RMSE 不看残差 | 一个数不能反映全部——可能存在系统偏差 | 阶段 4 必须做残差分析 |
| 认为越小越好的参数是最优的 | 过低的 RMSE 可能是过拟合 | 检查参数是否接近边界值（如 cps=0.001） |
| 用调参的 CV 结果来汇报精度 | 乐观偏差——你在选择最好的结果 | 保留一份完全未触碰的测试集 |
| 忽略参数交互 | 单独调每个参数，找不到真正最优 | 热力图联合搜索 |

---

## 9.9 核心概念清单

| 概念 | 一句话理解 |
|------|-----------|
| **分阶段调参** | 先定大方向（mode）→ 再调核心旋钮（cps×sps）→ 最后微调细节 |
| **参数交互** | 趋势和季节在"抢地盘"——需要协同调整，不能单独拧 |
| **热力图形态** | 单一低谷（数据清晰）vs 平坦区域（噪声大）vs 对角线（在抢地盘） |
| **乐观偏差** | 调参阶段看到的 RMSE 偏乐观——必须在独立数据上验证 |
| **参数接近边界** | cps=0.001 → 模型可能过于僵硬，需要检查是否真的需要这么紧 |

---

下一章进入数据预处理与异常处理——如何在实际建模前把数据准备好。
