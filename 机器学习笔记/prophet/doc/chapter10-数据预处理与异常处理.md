# 第 10 章：数据预处理与异常处理

理论上的数据是干净的 CSV，现实中的数据是脏的。这一章讲在 `model.fit(df)` 之前必须做的事。

---

## 10.1 缺失值：Prophet 能做什么、不能做什么

### 10.1.1 Prophet 对缺失值的天然容忍

因为 Prophet 是基于概率模型的（第 1 章 GAM 的"M"），它在缺失值处不需要插值——后验分布中的先验成分会在缺失处提供平滑过渡。

```python
# Prophet 天然容忍缺失——不需要你做任何事
df = pd.DataFrame({
    'ds': pd.date_range('2020-01-01', '2024-12-31', freq='D'),
    'y': ...  # 某些天的 y 是 NaN
})
# Prophet 自动处理，不会报错
model.fit(df)
```

### 10.1.2 但这不代表可以不管缺失值

| 缺失类型 | Prophet 行为 | 风险 | 建议 |
|----------|-------------|------|------|
| **随机零星缺失** (< 5%) | 先验平滑过渡，影响极小 | 无 | 不需要处理 |
| **连续大段缺失** (> 2 周) | 先验主导该段，趋势和季节估计不准确 | 中等 | 考虑是否该排除该时间段 |
| **季节关键节点缺失** | 季节性模式估计偏差 | 高 | 必须检查——比如连续缺了 12 月的数据，年季节不可能准 |
| **最近数据缺失** | 趋势末端估计不可靠 | 高 | 最近数据最重要——考虑缩短预测 horizon |

### 10.1.3 缺失值诊断代码

```python
import pandas as pd
import matplotlib.pyplot as plt

# 1. 检查缺失分布
df['ds'] = pd.to_datetime(df['ds'])
df = df.set_index('ds').sort_index()

# 补全日期范围，标记缺失
full_range = pd.date_range(df.index.min(), df.index.max(), freq='D')
missing_dates = full_range.difference(df.index)
print(f"缺失天数: {len(missing_dates)} / {len(full_range)} ({len(missing_dates)/len(full_range)*100:.1f}%)")

# 2. 连续缺失段落
if len(missing_dates) > 0:
    # 找连续缺失段
    gaps = (pd.Series(missing_dates).diff() > pd.Timedelta(days=1)).cumsum()
    for gap_id, group in pd.Series(missing_dates).groupby(gaps):
        print(f"  缺失段 {gap_id}: {group.iloc[0].date()} ~ {group.iloc[-1].date()} ({len(group)} 天)")

# 3. 可视化缺失
df['missing'] = 0
df.reindex(full_range)['missing'].fillna(1).plot(
    kind='line', figsize=(14, 2), lw=0, marker='|', markersize=5
)
plt.title('数据完整性检查（竖线 = 缺失日）')
plt.show()

df = df.reset_index()  # 恢复
```

---

## 10.2 异常值：检测 + 处理

### 10.2.1 异常值的两种类型

| 类型 | 示例 | 应该删除吗？ |
|------|------|-------------|
| **数据错误** | 系统 bug 导致某天销量 = 0，实际应该有数据 | ✅ 删除（它不是真实信号） |
| **真实极端事件** | 618 大促当天销量暴涨 10 倍 | ❌ 不删（它是真实信号，应该用节假日建模） |

**删除前先问自己**：如果明天再发生一次，我希望模型"看到"它吗？是 → 保留并用节假日/变点建模；否 → 删除或标记。

### 10.2.2 检测方法

```python
import numpy as np

# 方法 1: 统计阈值（IQR — 对非正态数据稳健）
def detect_outliers_iqr(series, k=3.0):
    """返回异常值的布尔掩码。k=1.5 是经典值，k=3.0 是保守值"""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (series < lower) | (series > upper)

# 方法 2: 滚动窗口（检测"突然跳跃"）
def detect_outliers_rolling(series, window=30, n_std=4):
    """检测偏离滚动均线超过 n_std 的点"""
    rolling_mean = series.rolling(window, center=True).mean()
    rolling_std = series.rolling(window, center=True).std()
    return np.abs(series - rolling_mean) > n_std * rolling_std

# 方法 3: 先用 Prophet 拟合再检测（利用预测区间）
model = Prophet()
model.fit(df)
forecast = model.predict(df[['ds']])
residuals = df['y'] - forecast['yhat']
outliers = np.abs(residuals) > 3 * residuals.std()

# 综合方法
df['outlier_iqr'] = detect_outliers_iqr(df['y'], k=3.0)
df['outlier_rolling'] = detect_outliers_rolling(df['y'])
df['outlier_prophet'] = outliers
df['is_outlier'] = df[['outlier_iqr', 'outlier_rolling', 'outlier_prophet']].sum(axis=1) >= 2

print(f"标记异常值: {df['is_outlier'].sum()} 个 / {len(df)} 总天数")
print("可疑日期:")
print(df[df['is_outlier']][['ds', 'y']].to_string())
```

### 10.2.3 处理策略

```python
# 策略 A: 直接删除（仅数据错误）
df_clean = df[~df['is_outlier']].copy()

# 策略 B: 替换为 NA——让 Prophet 的先验平滑过渡（推荐）
df_clean = df.copy()
df_clean.loc[df['is_outlier'], 'y'] = None  # None/NAN → Prophet 自动处理

# 策略 C: 用滚动中位数替换（如果异常值太多，不想全留空）
df_clean = df.copy()
rolling_median = df['y'].rolling(30, center=True, min_periods=5).median()
df_clean.loc[df['is_outlier'], 'y'] = rolling_median[df['is_outlier']]

# 策略 D: 保留但标记为特殊事件（用额外回归量）
# df['is_special_period'] = df['is_outlier'].astype(int)
# model.add_regressor('is_special_period')
```

---

## 10.3 数据聚合粒度

### 10.3.1 选择合适的时间粒度

| 粒度 | Prophet 适用？ | 何时用 |
|------|---------------|--------|
| **日 (D)** | ✅ 原生 | 大多数业务场景——零售、流量、运营指标 |
| **周 (W)** | ✅ | 以周为决策周期的业务——周报、周预算 |
| **月 (MS)** | ✅ | 月度规划、财务预测 |
| **季 (Q)** | ✅ | 长期战略预测 |
| **时 (H)** | ⚠️ 可用但不推荐 | Prophet 没有内置日内季节性的优秀模型 |
| **分/秒** | ❌ | 数据量巨大 + 模式完全不同 → 用 ARIMA 或 DL |

### 10.3.2 聚合代码

```python
# 日 → 周
df_weekly = df.set_index('ds').resample('W').agg({
    'y': 'sum',  # 销量 → 求和
    # 'y': 'mean',  # 温度 → 求均值
}).reset_index()

# 日 → 月
df_monthly = df.set_index('ds').resample('MS').sum().reset_index()

# ⚠️ 聚合后 make_future_dataframe 的 freq 需要匹配
model = Prophet()
model.fit(df_weekly)
future = model.make_future_dataframe(periods=52, freq='W')  # 不是 'D'！
```

### 10.3.3 粒度选择的权衡

```
细粒度（日）
  ✅ 更多数据点 → 参数估计更准
  ✅ 可以捕捉周末效应等精细季节模式
  ❌ 噪声大
  ❌ 计算更慢（交叉验证 cutoff 更多）

粗粒度（周）
  ✅ 噪声被平滑
  ✅ 计算更快
  ❌ 丢失周末效应
  ❌ 数据点少 → 季节估计不牢靠
```

> **经验法则**：如果周末波动对你的业务判断不重要，且数据噪声大，升到周粒度往往能提高预测准确性。但如果周末/工作日差异是核心关注点，保持日粒度。

---

## 10.4 数据质量检查清单

在 `model.fit(df)` 之前，跑一遍这个清单：

```python
def data_quality_report(df):
    """在建模前生成数据质量报告"""
    df = df.copy()
    df['ds'] = pd.to_datetime(df['ds'])

    print("=" * 50)
    print("数据质量报告")
    print("=" * 50)

    # 1. 基本形状
    print(f"\n1. 数据量: {len(df)} 行")
    print(f"   日期范围: {df['ds'].min().date()} ~ {df['ds'].max().date()}")
    print(f"   时间跨度: {(df['ds'].max() - df['ds'].min()).days} 天")

    # 2. 缺失检查
    full_days = pd.date_range(df['ds'].min(), df['ds'].max(), freq='D')
    missing = len(full_days) - len(df)
    print(f"\n2. 缺失天数: {missing} / {len(full_days)} ({missing/len(full_days)*100:.1f}%)")

    # 3. 数值检查
    print(f"\n3. y 统计:")
    print(f"   Min: {df['y'].min():.2f}")
    print(f"   Max: {df['y'].max():.2f}")
    print(f"   Mean: {df['y'].mean():.2f}")
    print(f"   Std: {df['y'].std():.2f}")
    print(f"   负值数: {(df['y'] < 0).sum()} (应为 0)")
    print(f"   零值数: {(df['y'] == 0).sum()}")

    # 4. 异常值
    q1, q3 = df['y'].quantile(0.25), df['y'].quantile(0.75)
    iqr = q3 - q1
    n_outliers = ((df['y'] < q1 - 3*iqr) | (df['y'] > q3 + 3*iqr)).sum()
    print(f"\n4. 异常值 (3×IQR): {n_outliers} ({n_outliers/len(df)*100:.1f}%)")
    if n_outliers > len(df) * 0.05:
        print("   ⚠️ 异常值占比 > 5%，建议排查")

    # 5. 趋势存在性
    if len(df) > 30:
        half = len(df) // 2
        first_half_mean = df['y'].iloc[:half].mean()
        second_half_mean = df['y'].iloc[half:].mean()
        change_pct = (second_half_mean - first_half_mean) / first_half_mean * 100
        print(f"\n5. 趋势检查:")
        print(f"   前半段均值: {first_half_mean:.1f}")
        print(f"   后半段均值: {second_half_mean:.1f}")
        print(f"   变化: {change_pct:.1f}%")

    # 6. 季节性检查
    df['weekday'] = df['ds'].dt.dayofweek
    df['month'] = df['ds'].dt.month
    print(f"\n6. 季节性初步检查:")
    print(f"   周内变异系数: {df.groupby('weekday')['y'].mean().std() / df['y'].mean():.3f}")
    print(f"   月间变异系数: {df.groupby('month')['y'].mean().std() / df['y'].mean():.3f}")
    print(f"   (值 > 0.1 通常意味着存在该周期性)")

    # 7. 建议
    print(f"\n7. 建模建议:")
    total_days = (df['ds'].max() - df['ds'].min()).days
    if total_days < 365:
        print("   ⚠️ 数据不足 1 年，年季节性不可靠")
    if missing > len(full_days) * 0.2:
        print("   ⚠️ 缺失 > 20%，检查数据采集管道")
    if (df['y'] < 0).sum() > 0:
        print("   ⚠️ 存在负值 — 如果 y 是非负指标，需要排查")
    if df['y'].max() / (df['y'].min() + 1) > 10:
        print("   💡 数值跨度 > 10 倍，考虑乘性季节性")

    print("=" * 50)

# 使用
data_quality_report(df)
```

---

## 10.5 特殊场景处理

### 10.5.1 COVID 时期数据

如果数据包含 2020 年，那年的模式可能完全不具备代表性。处理策略：

```python
# 选项 A: 去掉异常时期（推荐——如果你有足够的其他年份数据）
df_clean = df[~df['ds'].between('2020-02-01', '2020-06-30')]

# 选项 B: 保留但标记为一个"超级节假日"
lockdown = pd.DataFrame({
    'holiday': 'covid_lockdown',
    'ds': pd.to_datetime(['2020-02-01']),
    'lower_window': -7,
    'upper_window': 150,  # 整个封锁期
})
model = Prophet(holidays=lockdown, holidays_prior_scale=50.0)

# 选项 C: 在封锁期前后各放一个变点
model = Prophet(changepoints=['2020-02-01', '2020-07-01'])
```

### 10.5.2 零点/零值处理

```python
# 场景：每日销量 = 0 的天数是真实的（比如周日门店关门）
# 不删！这些 0 是正确信号

# 但如果 0 导致 MAPE 炸裂：
# 在计算 MAPE 时过滤掉 y=0 的点
def safe_mape(y_true, y_pred):
    mask = y_true != 0
    return (np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])).mean() * 100

# 或在长期评估中用 SMAPE 替代 MAPE
```

### 10.5.3 趋势断裂

```python
# 场景：公司业务模式在某个时间点发生了根本变化（如收购、转型）
# 之前的模式和之后的模式完全不可比
# 策略：只用断裂后的数据

break_date = '2023-06-01'
df_new_era = df[df['ds'] >= break_date].copy()

# 如果断裂后数据太少，可以在断裂点加一个强变点
model = Prophet(
    changepoints=[break_date],
    changepoint_prior_scale=1.0,  # 给这个大转折足够的灵活性
)
```

---

## 10.6 常见问题速查

| 问题 | 表现 | 解决 |
|------|------|------|
| 大量连续 0 值 | 季节性被压扁 | 单独建模有数据的子集，或聚合到周粒度 |
| 年底突然掉到 0 | 数据采集中断，不是真实信号 | 标记为 NA |
| 量级跨越了几个数量级 | 加性模型效果差 | 对 y 取 log 后用加性模式，或直接切换乘性模式 |
| 日期列包含时区 | Windows 上可能报 warning | `df['ds'] = df['ds'].dt.tz_localize(None)` |
| 周末和节假日的 0 值含义不同 | 前者是真实关门，后者是数据缺失 | 需要对缺失原因做标记 |

---

## 10.7 核心概念清单

| 概念 | 一句话理解 |
|------|-----------|
| **Prophet 的缺失容忍** | 概率模型天然处理 NA，不需要插值——但大段缺失仍不可接受 |
| **异常值先问来源** | 数据错误 → 删；真实事件 → 用节假日/变点建模 |
| **粒度的权衡** | 细 = 多信息多噪声；粗 = 平滑但丢细节 |
| **质量检查清单** | `fit` 前必跑的 7 项检查——防止垃圾进垃圾出 |
| **COVID/特殊时期** | 要么删，要么标记为"超级事件"，不能假装不存在 |

---

下一章进入大规模预测——当你有几百条时间序列需要同时建模时怎么办。
