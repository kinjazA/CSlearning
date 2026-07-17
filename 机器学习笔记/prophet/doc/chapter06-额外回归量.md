# 第 6 章：额外回归量

前几章建模的都是序列**自身**的结构——趋势、季节、节假日。但现实中的预测往往受**外部因素**影响：降价促销拉高销量、高温天气推升用电、竞品上线压低流量。这些外部信息也应该进入模型——Prophet 通过**额外回归量 (Additional Regressors)** 来实现。

---

## 6.1 额外回归量 vs 节假日：一个框架，两种形态

两者在 Prophet 内部用的是同一套机制——都是 GAM 的附加项。区别在于取值类型：

| 特征 | 节假日 | 额外回归量 |
|------|--------|-----------|
| **取值** | 0/1（该天是否属于节假日窗口） | **任意连续值** |
| **典型用途** | 促销日、法定假日、特殊事件日 | 价格、温度、广告投放金额、竞品指数 |
| **未来值** | 天然已知（促销计划、法定日历） | **需要你提供**——这是核心挑战 |
| **先验** | `holidays_prior_scale` | 每个回归量独立的 `prior_scale` |

> 从数学上看：节假日本质上是额外回归量的一个特例——取值只有 0 和 1。

---

## 6.2 加性 vs 乘性回归量

和季节性一样，额外回归量也分加性和乘性：

```python
# 加性（默认）：y = ... + β × regressor_value
model.add_regressor('price', mode='additive')

# 乘性：y = ... × (1 + β × regressor_value)
model.add_regressor('temperature', mode='multiplicative')
```

### 选择指南

| 场景 | 建议 | 理由 |
|------|------|------|
| 价格变化带来固定的销量变化（降 10 元多卖 50 件） | 加性 | 绝对效应恒定 |
| 广告投入的回报随业务体量同比放大 | 乘性 | 效应对数线性 |
| 温度对用电量的影响——冬夏都是极端天气用更多 | 注意 | 可能需要非线性的 U 形处理（见 6.5） |

---

## 6.3 代码实操

### 6.3.1 基本用法

```python
import pandas as pd
from prophet import Prophet

# ============================================================
# 场景：预测餐厅日营收，外部变量 = 当日是否有促销 + 广告投放金额
# ============================================================

df = pd.read_csv('restaurant_sales.csv')
df['ds'] = pd.to_datetime(df['ds'])

# Step 1: 创建模型并注册回归量
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
)

# add_regressor 必须在 fit 之前调用！
model.add_regressor('promo', mode='additive')          # 促销标志（0/1）
model.add_regressor('ad_spend', mode='additive')       # 广告投放金额（元）

model.fit(df)
```

### 6.3.2 关键步骤：提供未来值

这是额外回归量使用中**最核心、最容易被忽略**的一步。`make_future_dataframe` 只能生成 `ds` 列，**你必须手动补充回归量在未来日期的值**：

```python
# Step 2: 创建未来 DataFrame
future = model.make_future_dataframe(periods=30)

# Step 3: ⚠️ 必须手动填充回归量的未来值！
# 如果这一行漏了，predict 会报错或产生错误预测

# 假设未来 30 天每天都有促销
future['promo'] = 1
# 假设未来 30 天广告预算每天 5000 元
future['ad_spend'] = 5000

# 也可以精细到每天不同：
# future.loc[future['ds'].dt.dayofweek < 5, 'ad_spend'] = 5000  # 工作日
# future.loc[future['ds'].dt.dayofweek >= 5, 'ad_spend'] = 3000 # 周末

forecast = model.predict(future)
```

### 6.3.3 查看回归量的贡献

```python
# 在 forecast 里看回归量的效应
forecast[['ds', 'promo', 'ad_spend']].tail()

# 成分分解图中也会出现
model.plot_components(forecast)
```

### 6.3.4 控制每个回归量的先验

```python
model.add_regressor('promo', prior_scale=5.0)        # 促销影响较确定，偏保守
model.add_regressor('ad_spend', prior_scale=20.0)     # 广告 ROI 波动大，给更多灵活性
model.add_regressor('temperature', prior_scale=15.0, mode='multiplicative')
```

和变点/季节性/节假日一样：`prior_scale` 小 = 不相信回归量有太大影响（向 0 收缩），`prior_scale` 大 = 让数据充分决定回归量的效应。

---

## 6.4 回归量的第一道门槛：未来值拿得到吗

额外回归量使用中**最重要的一条铁律**：

> **再加一个回归量之前，先问自己：这个变量在未来的值我能拿到吗？如果答案是"不能"，不管它历史上和 y 有多强的相关性，都不要加。**

这个约束是额外的回归量和趋势/季节/节假日最大的不同。趋势、季节、节假日——Prophet 会自动推算它们的未来值。但回归量完全是你提供的，Prophet 不会替你"猜"它未来是多少。

### 判定流程

```
候选回归量
    │
    ├── 未来值天然已知？
    │   └── 促销计划、法定节假日、已排期的活动
    │       → ✅ 直接加入
    │
    ├── 未来值可以预测？
    │   └── 天气预报、宏观经济指标（有第三方预测）、
    │       周期性变量（可用历史均值）
    │       → ⚠️ 可以加，但要接受预测误差会传导
    │
    ├── 未来值可以人工设定？
    │   └── 广告预算、价格策略（你自己控制的）
    │       → ✅ 多场景模拟——不同预算对应不同预测
    │
    └── 未来值完全不可知？
        └── 竞品定价、突发热点事件、股市指数
            → ❌ 不要加。历史再相关也没有用。
```

### 一个翻车案例

```python
# ❌ 反模式：加了一个历史上高度相关、但未来完全不可知的回归量
df['stock_index'] = get_stock_data()  # 历史上和销量高度相关
model.add_regressor('stock_index')

# 预测时——你填什么？
future['stock_index'] = ???  # 你不知道明天股市是多少
# 你只能瞎填一个值 → 这个回归量不仅没帮助，反而会破坏预测
```

```python
# ✅ 正确：只加未来可控或可预测的变量
model.add_regressor('promo')          # 促销——你的计划，你知道
model.add_regressor('ad_budget')      # 广告预算——你决定的
model.add_regressor('temperature')    # 温度——有天气预报
# 这三个未来值都能合理获得 → 可以加入
```

---

## 6.5 实战中的最大难题：未来值从哪来

额外回归量的未来值是**你的责任**，Prophet 不会帮你推算。这是工程上的核心挑战，也是很多项目翻车的地方。

### 策略一：已知未来值 → 直接填

```python
# 节假日促销计划已定
future['promo'] = future['ds'].isin(promo_plan).astype(int)

# 历史最低价策略
future['price'] = future['ds'].apply(get_planned_price)
```

### 策略二：可预测的未来值 → 单独建模

```python
# 用天气预报作为温度回归量的未来值
weather_forecast = pd.read_csv('weather_forecast_30days.csv')
future = future.merge(weather_forecast, on='ds', how='left')
future['temperature'].fillna(df['temperature'].mean(), inplace=True)  # fallback
```

### 策略三：周期性变量 → 用历史均值或 Prophet 自预测

```python
# 广告投放本身有周期，可以先用 Prophet 预测广告投放量
ad_model = Prophet()
ad_model.fit(df[['ds', 'ad_spend']].rename(columns={'ad_spend': 'y'}))
ad_future = ad_model.predict(future[['ds']])
future['ad_spend'] = ad_future['yhat'].clip(lower=0)  # 广告费不能为负
```

### 策略四：不可知变量 → 场景模拟

对于完全不可知的未来值（如竞争对手定价），不做单一预测，而是做**多场景**：

```python
scenarios = {
    'optimistic':  {'competitor_price': 100},  # 竞品涨价
    'baseline':    {'competitor_price': 120},
    'pessimistic': {'competitor_price': 140},  # 竞品大降价
}

for name, values in scenarios.items():
    future_scenario = future.copy()
    future_scenario['competitor_price'] = values['competitor_price']
    forecast = model.predict(future_scenario)
    print(f"{name}: mean yhat = {forecast['yhat'].mean():.0f}")
```

### 策略五：滞后变量作为回归量（不推荐但常用）

```python
# 把昨天的销量作为回归量
# ⚠️ 风险：预测时会用预测值前一天的值，误差会累积
df['yesterday_sales'] = df['y'].shift(1)
model.add_regressor('yesterday_sales')

# 未来值的填充需要迭代进行——非常脆弱
```

---

## 6.6 非线性关系的处理

Prophet 的额外回归量是**线性**的（β × x），但如果真实关系是非线性的呢？

### 问题：温度的 U 形效应

空调用电：太冷不用、太热狂用 → U 形关系。一条直线根本拟合不了。

### 方案一：特征工程（推荐）

```python
# 手动构造多项式/分箱特征
df['temp_squared'] = df['temperature'] ** 2
df['temp_low'] = (df['temperature'] < 10).astype(float)   # 低温取暖
df['temp_high'] = (df['temperature'] > 30).astype(float)  # 高温制冷

model.add_regressor('temperature')
model.add_regressor('temp_squared')
model.add_regressor('temp_low')
model.add_regressor('temp_high')
# 通过这些组合，模型等价于拟合一条"分段抛物线"
```

### 方案二：交互项

```python
# 促销 + 周末 = 超加性效应
df['promo_X_weekend'] = df['promo'] * (df['ds'].dt.dayofweek >= 5).astype(float)
model.add_regressor('promo')
model.add_regressor('promo_X_weekend')
```

---

## 6.7 完整示例：餐厅日营收预测

```python
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

# ============================================================
# 数据准备
# ============================================================
np.random.seed(42)
dates = pd.date_range('2022-01-01', '2024-12-31', freq='D')
n = len(dates)

# 模拟真实关系:
# 营收 = 基线 + 趋势 + 季节性 + 促销效应 + 广告效应 + 天气效应 + 噪声
trend = np.linspace(8000, 12000, n)  # 缓慢增长
weekly = 2000 * np.sin(2 * np.pi * np.arange(n) / 7)  # 周末涨
yearly = 1500 * np.sin(2 * np.pi * np.arange(n) / 365.25)
promo_effect = np.where(np.arange(n) % 30 < 3, 3000, 0)  # 每月前3天促销
ad_effect = 0.3 * np.random.randn(n).cumsum() + 500  # 广告投放的累积效应
weather_effect = -abs(25 - (15 + 10 * np.sin(2 * np.pi * np.arange(n) / 365.25))) * 100  # 温度U形

df = pd.DataFrame({
    'ds': dates,
    'y': trend + weekly + yearly + promo_effect + ad_effect + weather_effect + np.random.randn(n) * 500,
    'promo': np.where(np.arange(n) % 30 < 3, 1, 0),
    'ad_spend': np.random.gamma(2, 200, n),  # 广告投放金额
    'temperature': 15 + 10 * np.sin(2 * np.pi * np.arange(n) / 365.25) + np.random.randn(n) * 2,
})

# 构造温度的非线性特征
df['temp_deviation'] = np.abs(df['temperature'] - 25)  # 偏离25度越远，影响越大

# 划分训练集（留最后90天做对比）
train = df.iloc[:-90].copy()
test = df.iloc[-90:].copy()[['ds', 'y', 'promo', 'ad_spend', 'temperature', 'temp_deviation']]

# ============================================================
# 建模
# ============================================================
model = Prophet(
    yearly_seasonality=15,
    weekly_seasonality=5,
    changepoint_prior_scale=0.05,
)

model.add_regressor('promo')
model.add_regressor('ad_spend')
model.add_regressor('temperature')
model.add_regressor('temp_deviation')

model.fit(train)

# ============================================================
# 预测（含未来值的填充）
# ============================================================
future = model.make_future_dataframe(periods=90)

# 填充已知的未来回归量
future = future.merge(
    test[['ds', 'promo', 'ad_spend', 'temperature', 'temp_deviation']],
    on='ds', how='left'
)

# 如果测试集覆盖不全（实际场景中未来值需要单独提供），填充默认值
future['promo'].fillna(0, inplace=True)
future['ad_spend'].fillna(future['ad_spend'].mean(), inplace=True)
future['temperature'].fillna(future['temperature'].mean(), inplace=True)
future['temp_deviation'].fillna(future['temp_deviation'].mean(), inplace=True)

forecast = model.predict(future)

# ============================================================
# 评估
# ============================================================
comparison = forecast[['ds', 'yhat']].merge(test[['ds', 'y']], on='ds')
comparison['error'] = comparison['y'] - comparison['yhat']

print(f"MAE:  {comparison['error'].abs().mean():.0f}")
print(f"MAPE: {(comparison['error'].abs() / comparison['y']).mean() * 100:.1f}%")

# ============================================================
# 可视化
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# 预测 vs 实际
ax = axes[0]
ax.plot(comparison['ds'], comparison['y'], label='Actual', alpha=0.7)
ax.plot(comparison['ds'], comparison['yhat'], label='Predicted', alpha=0.7)
ax.legend()
ax.set_title('预测 vs 实际（测试集 90 天）')

# 各回归量的贡献
forecast[['ds', 'promo', 'ad_spend', 'temperature', 'temp_deviation']].set_index('ds').plot(ax=axes[1])
axes[1].set_title('各回归量对预测的贡献')
axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()
```

---

## 6.8 常见问题速查

| 问题 | 原因 | 解决 |
|------|------|------|
| `ValueError: Regressor 'xxx' missing from dataframe` | 在 `add_regressor` 后该列不在 `future` 中 | `future['xxx'] = ...` 填充未来值 |
| 回归量效应近乎为零 | `prior_scale` 太小，先验压平了估计 | 调大该回归量的 `prior_scale` |
| 回归量系数的符号与业务直觉相反 | 遗漏了混淆变量（比如同时降价和加大广告） | 检查多重共线性，考虑去掉高度相关的回归量 |
| 预测的回归量贡献突然跳变 | 未来值填充用的默认值不合理 | 设计更合理的未来值估算逻辑 |
| 加入回归量后交叉验证反而变差 | 回归量过拟合，或未来值估计不准 | 减少回归量数量，或用更保守的 `prior_scale` |

---

## 6.9 核心概念清单

| 概念 | 一句话理解 |
|------|-----------|
| **额外回归量** | 你自己提供的外部变量，以线性方式加入 GAM |
| **未来值问题** | Prophet 不帮你推算回归量的未来值——这是你的工程责任 |
| **prior_scale** | 每个回归量独立的先验——控制"外部变量的影响有多大" |
| **非线性处理** | 通过特征工程（平方项、分箱、交互项）突破线性假设 |
| **场景模拟** | 对未来不可知变量做多组预测——乐观/基准/悲观 |

### 参数速查

| 操作 | 代码 | 注意 |
|------|------|------|
| 注册回归量 | `model.add_regressor(name, prior_scale, mode)` | 必须在 `fit` 前调用 |
| 填充未来值 | `future['name'] = values` | 最容易漏的一步 |
| 查看效应 | `forecast[['ds', 'regressor_name']]` | 列名即注册时的 name |

---

下一章讲不确定性量化——`yhat_lower` 和 `yhat_upper` 是怎么算出来的，什么时候该用 MCMC。
