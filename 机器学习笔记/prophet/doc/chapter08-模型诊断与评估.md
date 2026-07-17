# 第 8 章：模型诊断与评估

前面七章你学会了建模。这一章回答一个问题：**这个模型到底好不好？**

好不是"看起来拟合得不错"——需要量化的诊断框架。

---

## 8.1 时间序列不能随机切

### 8.1.1 为什么标准交叉验证不适用

机器学习中，你习惯这样划分数据：

```
标准 K-Fold CV（对时间序列是灾难）：
  训练    测试    训练
  ┌────┐ ┌──┐ ┌────┐
  │····│·│  │·│····│   ← 用未来数据训练，预测过去——时间旅行
  └────┘ └──┘ └────┘
```

时间序列的**根本约束**：你不能用未来来预测过去。这意味着：

1. 训练数据必须在测试数据**之前**
2. 测试窗口之间不能重叠（严格来说，也不应该）
3. 一次训练，一次验证

### 8.1.2 时间序列交叉验证：滚动窗口

正确的做法是**单链滚动 (Rolling Forecast Origin / Expanding Window)**：

```
历史数据:  |····························|
                     │
         ┌───────────┼──────┐
Cutoff 1 │   训练     │ 测试 │
         └───────────┼──────┘
               ┌───────────┼──────┐
Cutoff 2       │   训练     │ 测试 │
               └───────────┼──────┘
                     ┌───────────┼──────┐
Cutoff 3             │   训练     │ 测试 │
                     └───────────┼──────┘

每次向前挪动 cutoff → 训练集扩大 → 在新的"未来"上评估
```

Prophet 内置了这套机制，API 很简洁。

---

## 8.2 Prophet 的交叉验证工具

### 8.2.1 核心参数

```python
from prophet.diagnostics import cross_validation

df_cv = cross_validation(
    model,
    initial='730 days',      # 初始训练集最小天数
    period='180 days',       # 每隔多久做一次 cutoff
    horizon='365 days',      # 每次预测多远
)
```

这三个参数控制了滚动的方式：

```
initial=730 days, period=180 days, horizon=365 days

  过去                        现在        未来
  |··························|···········|
  |←── 至少 730 天训练 ──→|← 365天预测 →|   Cutoff 1
        |←── 至少 730 天训练 ──→|← 365 →|   Cutoff 2
              |←── 至少 730 天训练 ──→|← 365 →|   Cutoff 3
              ↑                ↑
          每隔 180 天          每次向前预测 365 天
          做一次 cutoff
```

| 参数 | 含义 | 选择建议 |
|------|------|----------|
| `initial` | 初始训练集最少多少天 | ≥ 3 个完整季节周期（如年季节 → ≥ 1095 天） |
| `period` | cutoff 之间的间隔 | 根据数据密度：日数据 30-180 天，月数据 3-6 个月 |
| `horizon` | 每次预测多看远 | 和你的业务预测 horizon 一致 |

> ⚠️ 注意：`initial` 是**最少**训练量。实际训练集会随 cutoff 前移而扩大（expanding window），除非你指定 `'horizon'` 参数的配套行为。

### 8.2.2 代码

```python
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation

# 1. 拟合模型
model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
model.fit(df)

# 2. 交叉验证
df_cv = cross_validation(
    model,
    initial='1095 days',    # 至少 3 年训练数据
    period='180 days',      # 每半年评估一次
    horizon='365 days',     # 预测一年
)

# df_cv 的结构
print(df_cv.columns)
# Index(['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'y', 'cutoff'])
#       日期    预测值    下界         上界        真实值   哪个cutoff

print(f"共 {df_cv['cutoff'].nunique()} 个 cutoff, {len(df_cv)} 条预测记录")
```

### 8.2.3 查看每次 cutoff 的预测表现

```python
# 按 cutoff 分组看预测精度
for cutoff, group in df_cv.groupby('cutoff'):
    errors = group['y'] - group['yhat']
    mae = errors.abs().mean()
    print(f"cutoff={cutoff.date()}: MAE={mae:.1f}, 样本数={len(group)}")
```

---

## 8.3 预测误差指标

Prophet 自带 `performance_metrics` 工具，一次性输出常用指标：

```python
from prophet.diagnostics import performance_metrics

df_metrics = performance_metrics(df_cv)
print(df_metrics.head())
```

### 8.3.1 各指标的含义与适用场景

| 指标 | 公式 | 量纲 | 何时关注 |
|------|------|------|----------|
| **MSE** | $\frac{1}{n}\sum (y_i - \hat{y}_i)^2$ | 原值² | 对大误差敏感——惩罚离群预测 |
| **RMSE** | $\sqrt{\text{MSE}}$ | 原值 | **最常用**——与原值同量纲，直观 |
| **MAE** | $\frac{1}{n}\sum \|y_i - \hat{y}_i\|$ | 原值 | 对离群值不敏感——稳健 |
| **MAPE** | $\frac{1}{n}\sum \|\frac{y_i - \hat{y}_i}{y_i}\| \times 100\%$ | % | 跨序列比较——但 $y$ 接近 0 时炸裂 |
| **SMAPE** | $\frac{1}{n}\sum \frac{\|y_i - \hat{y}_i\|}{(\|y_i\| + \|\hat{y}_i\|)/2} \times 100\%$ | % | MAPE 的改进版——有界（0-200%） |
| **MDAPE** | 绝对百分比误差的**中位数** | % | 比 MAPE 更稳健——不受极端值影响 |

### 8.3.2 选择指南

```
你需要跨品类比较预测精度吗？
  ├── 是 → 各类销量量级不同 → 用 MAPE 或 SMAPE
  └── 否 → 用 RMSE（直观，对大误差敏感）

你的数据中有接近 0 的天数吗？
  ├── 是 → 避免 MAPE（除以零或爆炸）→ 用 SMAPE 或 MAE
  └── 否 → MAPE 也可以

你需要向上汇报吗？
  └── 是 → MAPE / SMAPE（百分比，业务方容易理解）
```

### 8.3.3 代码

```python
from prophet.diagnostics import performance_metrics
import numpy as np

# Prophet 内置
df_metrics = performance_metrics(df_cv)
print(df_metrics[['horizon', 'mse', 'rmse', 'mae', 'mape', 'mdape']].to_string())

# 按预测 horizon 看精度衰减
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, metric in zip(axes.flat, ['mse', 'rmse', 'mae', 'mape']):
    ax.plot(df_metrics['horizon'], df_metrics[metric], 'o-')
    ax.set_title(metric.upper())
    ax.set_xlabel('Horizon (天)')
    ax.set_ylabel(metric.upper())
plt.tight_layout()
plt.show()

# 手动计算——如果你需要非标准指标
def mase(y_true, y_pred, y_train, seasonality=1):
    """Mean Absolute Scaled Error — 与朴素预测对比"""
    naive_error = np.abs(np.diff(y_train, seasonality)).mean()
    return np.abs(y_true - y_pred).mean() / naive_error

# 在交叉验证结果上计算
cv_errors = df_cv['y'] - df_cv['yhat']
print(f"RMSE: {np.sqrt((cv_errors ** 2).mean()):.2f}")
print(f"MAE:  {np.abs(cv_errors).mean():.2f}")
```

---

## 8.4 按预测 horizon 分析误差

交叉验证结果 `df_cv` 里，每一行都是一次具体的预测。但有些行预测的是"明天"，有些是"365 天后"——它们的难度完全不同。**按 horizon 分析就是问：我的模型预测 1 天后有多准？7 天后呢？365 天后呢？**

---

### 8.4.1 什么是 horizon

在交叉验证中，horizon = **预测日期距离 cutoff 的天数**。

```
cutoff = 2023-01-01, horizon = 365 天

  cutoff          horizon=1      horizon=30      horizon=365
    │                │               │               │
    ▼                ▼               ▼               ▼
────┼────────────────┼───────────────┼───────────────┼────
    │ 训练数据        │  预测 D+1      │ 预测 D+30     │预测 D+365
    │ (到 2023-01-01) │ "明天多少"     │ "下个月多少"   │ "一年后多少"
```

一个 cutoff 产生的 `df_cv` 行里，`horizon` 从 1 天一直变到 365 天。三个 cutoff 就会有三组"1 天预测"、三组"2 天预测"……以此类推。

```python
# 直接看 df_cv 里的 horizon 列
print(df_cv[['ds', 'cutoff', 'y', 'yhat']].head(10))

# 可以看到同一个 cutoff 下，ds 逐渐远离 cutoff →
# horizon 逐渐增大
df_cv['horizon'] = (df_cv['ds'] - df_cv['cutoff']).dt.days
print(df_cv[['cutoff', 'ds', 'horizon', 'y', 'yhat']].head(10))
```

典型输出：

```
   cutoff      ds          horizon  y       yhat
   2022-01-01  2022-01-02  1        120     118     ← 预测明天，很准
   2022-01-01  2022-01-03  2        115     119     ← 预测后天，误差开始出现
   2022-01-01  2022-01-04  3        130     122
   ...
   2022-01-01  2023-01-01  365      180     210     ← 预测一年后，误差大
```

---

### 8.4.2 为什么要按 horizon 分组看

如果不分组，你把"预测明天"和"预测一年后"的误差混在一起算平均——这没有意义。预测明天的误差天然小，预测一年后的误差天然大。混在一起，你既不知道模型短期有多准，也不知道长期什么时候开始崩。

**分组的目的：画出"误差 vs horizon"曲线，找到模型的"保质期"。**

```
RMSE
  │
  │               ╱
  │            ╱
  │         ╱
  │      ╱
  │   ╱
  │╱
  └────────────────── horizon (天)
   1   30   90   180   365

"保质期" ≈ 90 天——之后误差加速增长
```

---

### 8.4.3 `performance_metrics` 怎么分组的——用具体数字走一遍

这个"窗口"是在 **horizon 维度上**滚动的，不是在时间维度上。用一个具体例子来说清楚。

---

#### 数据长什么样

假设你做了交叉验证，得到了 3 个 cutoff，每个 cutoff 预测未来 10 天。`df_cv` 长这样：

```
cutoff        ds          horizon  y     yhat   error
2023-01-01    2023-01-02  1        100   102    +2
2023-01-01    2023-01-03  2        105   101    -4
2023-01-01    2023-01-04  3        108   110    +2
...
2023-01-01    2023-01-11  10       120   115    -5

2023-07-01    2023-07-02  1        130   128    -2
2023-07-01    2023-07-03  2        135   132    -3
...
2023-07-01    2023-07-11  10       150   145    -5

2024-01-01    2024-01-02  1        110   113    +3
...
2024-01-01    2024-01-11  10       140   135    -5
```

关键是：**horizon=1 的预测有 3 行**（每个 cutoff 一行），horizon=2 的也有 3 行，以此类推。

---

#### `rolling_window=0`：一个 horizon 值一个点

不做平滑。horizon=1 的 3 行算一个 RMSE，horizon=2 的 3 行算一个 RMSE……每个 horizon 独立计算。

```
horizon  RMSE
   1      2.4   ← 只用 3 个 horizon=1 的 error 算出来的
   2      3.1   ← 只用 3 个 horizon=2 的 error
   3      2.8
  ...
  10      5.0

结果: 10 个数据点，锯齿状，因为每个点只有 3 个样本
```

---

#### `rolling_window=0.3`：用一个具体例子走一遍

`rolling_window=0.3` 意味着窗口大小 = 总 horizon 范围的 30%。总 horizon 范围是 1~10 = 9 天，30% ≈ 3 天。

Prophet 不是"每 3 天一段，互不重叠"——它是**滑动**的：

```
第 1 个窗口: horizon ∈ [1, 3]
  取 horizon=1,2,3 的所有行 → 3×3=9 条 → 算一个 RMSE
  标记这个 RMSE 对应中间位置: horizon≈2

第 2 个窗口: horizon ∈ [2, 4]    ← 向右滑动 1
  取 horizon=2,3,4 的所有行 → 9 条 → 算一个 RMSE
  标记: horizon≈3

第 3 个窗口: horizon ∈ [3, 5]    ← 再滑 1
  ...

...依此类推，直到覆盖整个 horizon 范围
```

最终输出一个 DataFrame：

```
horizon  rmse     mae      mape
  2      2.7      2.2      2.1%
  3      3.0      2.5      2.3%
  4      3.3      2.7      2.5%
  5      3.5      2.9      2.6%
 ...
  9      4.8      4.0      3.8%
```

**每一行 = 相邻 horizon 值共享样本的一次汇总**。因为相邻窗口之间有大量重叠（horizon=3 的样本同时出现在窗口 1、2、3 中），曲线是平滑的。

---

#### `rolling_window` 大小的影响

```python
rolling_window = 0（不平滑）
  窗口大小 = 1 天 → 每个点样本少 → 锯齿多 → 能看到细节但噪声大
  适合：数据量大、想看精细结构

rolling_window = 0.1（默认）
  窗口大小 = horizon 总范围的 10%
  例如 horizon 范围 365 天 → 窗口 ~36 天 → 适度平滑
  适合：日常使用

rolling_window = 0.3（高度平滑）
  窗口大小 = 30% → 窗口 ~110 天 → 非常平滑
  适合：只看整体趋势，不在乎局部波动
```

---

#### 拿到这个序列后怎么分析

返回的 DataFrame 就是一条"误差-vs-horizon"曲线。分析它就是在回答三个问题：

**问题 1：曲线整体往上走多快？**

```
平缓上升（健康）:           陡峭上升（警告）:
RMSE                         RMSE
  │          ╱                 │              ╱
  │       ╱╱                   │           ╱╱
  │    ╱╱                      │        ╱╱
  │ ╱╱                         │    ╱╱╱
  │╱                           │╱╱╱
  └───────────── horizon       └───────────── horizon
  
  → 模型在远距离仍可用          → 模型很快失效，不适合长期预测
```

怎么量化：计算 RMSE 在整条曲线上的增长率——`(最后一点的 RMSE - 第一点的 RMSE) / 第一点的 RMSE`。如果比值 > 2（即翻了一倍以上），模型在远距离已经不太可靠。

**问题 2：有没有突然跳跃？**

```python
# 找 RMSE 在一阶差分上的突变
df_metrics['rmse_diff'] = df_metrics['rmse'].diff()
jump_horizons = df_metrics[df_metrics['rmse_diff'] > df_metrics['rmse_diff'].std() * 2]
# 如果有 → 在那些 horizon 附近，模型突然失准
# 原因可能是：季节性周期（如 7 天、30 天、180 天）在那个距离开始错位
```

**问题 3：MAPE 什么时候超过你的容忍线？**

```python
# 你的业务容忍度
max_mape = 20  # 最多接受 20% 的误差

# 找到 MAPE 首次超过 20% 的 horizon
exceed = df_metrics[df_metrics['mape'] > max_mape]
if len(exceed) > 0:
    safe_horizon = exceed['horizon'].min()
    print(f"模型在 horizon > {safe_horizon:.0f} 天后不可靠")
else:
    print(f"模型在整个预测范围内 MAPE < {max_mape}%")
```

---

### 8.4.4 如何读懂 horizon 曲线

曲线有几种典型形态，每种对应不同的模型问题：

```
形态 1: 健康模型
RMSE
  │            ╱
  │         ╱╱
  │      ╱╱
  │   ╱╱
  │╱╱
  └────────────────── horizon
  误差缓慢增长，没有突然跳跃
  → 模型在预测范围内表现一致

形态 2: 断崖式崩坏
RMSE
  │                 │╲
  │                 │ ╲
  │                 │  ╲
  │╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱│
  └────────────────── horizon
  在某个 horizon 突然暴涨
  → 季节性错配（年季节在半年处最弱？周季节在 3-4 天后？
     某个周期成分在特定距离失效）

形态 3: 全程高误差
RMSE
  │ ╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱
  │╱              ╲
  │
  └────────────────── horizon
  误差从头到尾都很大，波动剧烈
  → 模型根本没学到东西——检查数据质量、是否遗漏关键特征
```

---

### 8.4.5 结合业务决策

horizon 分析最终要回答一个业务问题：**这个模型能用来做多远的决策？**

```python
# 设定你的"最大可接受误差"
max_acceptable_mape = 20  # 比如 20% MAPE

# 从 horizon 曲线中找到"保质期"
horizon_limit = df_metrics[df_metrics['mape'] < max_acceptable_mape]['horizon'].max()
print(f"在 MAPE < {max_acceptable_mape}% 的条件下, 最长可预测 {horizon_limit:.0f} 天")

# 决策：
# 如果 horizon_limit < 你的业务需求 → 要么改进模型，要么接受更大误差
# 如果 horizon_limit > 你的业务需求 → 模型够用
```

---

### 8.4.6 代码

```python
from prophet.diagnostics import performance_metrics
import matplotlib.pyplot as plt

# 按 horizon 分桶
df_metrics = performance_metrics(df_cv, rolling_window=0.1)

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, metric in zip(axes.flat, ['rmse', 'mae', 'mape', 'mdape']):
    ax.plot(df_metrics['horizon'], df_metrics[metric], 'o-', markersize=4)
    ax.set_title(f'{metric.upper()} vs Horizon')
    ax.set_xlabel('Horizon (天)')
    ax.set_ylabel(metric.upper())
    ax.grid(True, alpha=0.3)

    # 标注"保质期"——误差开始加速增长的点
    # 简单方法：找 RMSE 二阶差分最大的位置
    if metric == 'rmse':
        d2 = df_metrics['rmse'].diff().diff()
        if d2.max() > 0:
            knee = df_metrics['horizon'].iloc[d2.idxmax()]
            ax.axvline(x=knee, color='red', linestyle='--', alpha=0.5,
                       label=f'拐点 ~{knee:.0f}天')
            ax.legend()

plt.suptitle('预测精度随 Horizon 的衰减', fontsize=14)
plt.tight_layout()
plt.show()
```

---

## 8.5 残差分析

误差指标告诉你"平均错多少"，残差分析告诉你**"在哪里错了、怎么错的"**。

### 8.5.1 残差时序图

```python
# 残差 = 真实值 - 预测值
df_cv['residual'] = df_cv['y'] - df_cv['yhat']

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# 1. 残差随时间的变化
axes[0].scatter(df_cv['ds'], df_cv['residual'], alpha=0.3, s=10)
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_title('残差时序图 — 检查是否有系统偏差')

# 2. 残差分布
axes[1].hist(df_cv['residual'], bins=50, edgecolor='white')
axes[1].axvline(x=0, color='r', linestyle='--')
axes[1].set_title('残差分布 — 应该大致对称、以 0 为中心')

# 3. 残差的周内模式
df_cv['weekday'] = df_cv['ds'].dt.dayofweek
df_cv.groupby('weekday')['residual'].mean().plot(kind='bar', ax=axes[2])
axes[2].axhline(y=0, color='r', linestyle='--')
axes[2].set_title('按星期几的平均残差 — 检查周季节性是否被充分建模')

plt.tight_layout()
plt.show()
```

### 8.5.2 诊断信号

| 残差表现 | 可能原因 | 行动 |
|----------|---------|------|
| 残差均值显著偏离 0 | 模型有系统偏差 | 检查趋势是否合理、是否遗漏增长因素 |
| 残差有明显的周内模式 | 周季节性欠拟合 | 增大 `weekly_seasonality` 阶数或 `seasonality_prior_scale` |
| 残差在特定月份偏大 | 年季节性欠拟合或有特殊事件 | 增大年季节性阶数，或添加自定义节假日 |
| 残差随时间放大 | 乘性模式被当成加性建模 | 切换 `seasonality_mode='multiplicative'` |
| 残差自相关（ACF 显著） | 模型遗漏了时间结构 | 检查是否缺少额外回归量或季节性成分 |

### 8.5.3 自相关检查

```python
from pandas.plotting import autocorrelation_plot

# 残差不应该有显著的自相关——如果有，说明模型漏掉了结构
autocorrelation_plot(df_cv['residual'].dropna())
plt.axhline(y=0.05, color='r', linestyle='--')
plt.axhline(y=-0.05, color='r', linestyle='--')
plt.title('残差自相关 — 应该在置信区间内')
plt.show()
```

---

## 8.6 预测覆盖度检查

除了误差大小，还要检查预测区间的**校准度**：

```python
# 理想情况下，80% 区间应该覆盖约 80% 的实际值
def check_coverage(df_cv, interval_width=0.80):
    covered = (df_cv['y'] >= df_cv['yhat_lower']) & \
              (df_cv['y'] <= df_cv['yhat_upper'])
    return covered.mean()

coverage = check_coverage(df_cv)
print(f"区间覆盖率: {coverage:.1%} (期望 ~80%)")

# 如果 coverage 远低于 80% → 模型过于自信 → 开启 MCMC 或调大参数
# 如果 coverage 远高于 80% → 模型过于保守 → 区间太宽，信息量低
```

---

## 8.7 完整诊断工作流

```python
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import matplotlib.pyplot as plt

# ============================================================
# Step 1: 建模
# ============================================================
model = Prophet(
    yearly_seasonality=15,
    weekly_seasonality=5,
    changepoint_prior_scale=0.05,
    seasonality_mode='multiplicative',
)
model.fit(df)

# ============================================================
# Step 2: 交叉验证
# ============================================================
df_cv = cross_validation(
    model,
    initial='1095 days',
    period='180 days',
    horizon='365 days',
)

# ============================================================
# Step 3: 指标计算
# ============================================================
df_metrics = performance_metrics(df_cv)
print("=== 预测精度指标 ===")
print(df_metrics[['horizon', 'rmse', 'mae', 'mape', 'mdape']].head(10))

# 按 horizon 汇总
print("\n=== 按预测距离汇总 ===")
summary = df_metrics.groupby(
    pd.cut(df_metrics['horizon'], bins=[0, 30, 90, 180, 365],
           labels=['1月内', '1-3月', '3-6月', '6-12月'])
)[['rmse', 'mae', 'mape']].mean()
print(summary)

# ============================================================
# Step 4: 残差诊断
# ============================================================
df_cv['residual'] = df_cv['y'] - df_cv['yhat']

print(f"\n=== 残差诊断 ===")
print(f"均值: {df_cv['residual'].mean():.2f} (应为 ~0)")
print(f"标准差: {df_cv['residual'].std():.2f}")
print(f"偏度: {df_cv['residual'].skew():.2f} (应为 ~0, 正=向上偏)")
print(f"区间覆盖率: {check_coverage(df_cv):.1%}")

# ============================================================
# Step 5: 决策
# ============================================================
if abs(df_cv['residual'].mean()) > df['y'].std() * 0.1:
    print("⚠️ 残差均值偏离——模型有系统偏差")
if check_coverage(df_cv) < 0.7:
    print("⚠️ 区间覆盖率过低——模型过于自信，考虑开启 MCMC")
if check_coverage(df_cv) > 0.95:
    print("⚠️ 区间覆盖率过高——区间太宽，信息量不足")
```

---

## 8.8 常见问题速查

| 问题 | 原因 | 解决 |
|------|------|------|
| CV 报内存不足 | 数据量大 + cutoff 太多 | 增大 `period` 减少 cutoff 数 |
| MAPE 无穷大或 NaN | 真实值 $y=0$ 导致除零 | 换用 SMAPE 或 MAE |
| 预测精度在特定季节崩溃 | 季节性建模不足 | 增大对应傅里叶阶数或切换乘性模式 |
| 残差有强烈自相关 | 遗漏时间结构 | 检查是否缺少额外回归量或周期成分 |
| 每条 cutoff 表现差异大 | 模型不稳定 | 增大 `initial` 确保训练数据充足 |

---

## 8.9 核心概念清单

| 概念 | 一句话理解 |
|------|-----------|
| **滚动交叉验证** | 始终用过去预测未来——不能随机打乱 |
| **RMSE** | 最通用的误差指标——与大误差同量纲 |
| **MAPE** | 百分比误差——跨序列可比，但 $y \approx 0$ 时炸 |
| **残差分析** | 不只是看"平均错多少"，而是看"在哪里错" |
| **区间覆盖率** | 80% 区间应该覆盖约 80% 的真实值——偏离说明校准有问题 |

---

下一章进入超参数调优——用交叉验证的结果来系统性地选择最优参数组合。
