# 第 4 章：季节性建模

季节性 $s(t)$ 是 Prophet 的第二个核心 GAM 组件，负责捕捉固定周期的规律性波动——周末销量涨、夏天用电高、深夜流量低。

---

## 4.1 傅里叶级数：Prophet 如何"看到"周期

### 4.1.1 为什么不用哑变量？

传统方法会用哑变量来处理季节性——比如对"星期几"用 6 个 0/1 虚拟变量。但这样每个日子被独立估计，相邻日子之间没有任何"平滑"约束。结果：周三和周四的效应可能差异很大，而实际上它们应该比较接近。

Prophet 采用 **傅里叶级数 (Fourier Series)** 来建模季节性：

$$s(t) = \sum_{n=1}^{N} \left( a_n \cos\left(\frac{2\pi n t}{P}\right) + b_n \sin\left(\frac{2\pi n t}{P}\right) \right)$$

| 符号 | 含义 |
|------|------|
| $P$ | 周期长度（年=365.25，周=7，日=1） |
| $N$ | **傅里叶阶数 (Fourier order)**——用多少对 sin/cos |
| $a_n, b_n$ | 各频率分量的系数，从数据中学习 |

### 4.1.2 直觉：用圆叠加来逼近任意形状

傅里叶级数的核心思想：**任何周期性的波形，都可以用一系列不同频率的正弦波叠加来逼近**。

```
N=1（最低频）:    ───╱╲───    只用一对 sin/cos，只能画出最简单的正弦波
N=3（中频）:     ──╱╲╱╲──   三对叠加，能刻画一般的山峰形状
N=10（高频）:    ─╱╲╱╲╱╲─  十对叠加，能刻画双峰、陡升缓降等复杂形态
N=20（过高）:    ╱╲╱╲╱╲╱╲  可能追着噪声跑，过拟合
```

**类比**：想象你在用橡皮泥捏一条曲线。N=1 只有一块大橡皮泥（只捏得出大弧形），N=10 有十块不同大小的小橡皮泥（能捏出更多细节）。

---

### 4.1.3 傅里叶阶数 N 的选择

N 越大 → 季节性模式越灵活 → 能刻画更细节的波动 → 但也越容易过拟合噪声。

| 季节性类型 | 默认 N | 含义 | 何时调大 |
|-----------|--------|------|----------|
| 年季节性 `yearly` | **10** | 用 10 对 sin/cos 拟合年度模式 | 数据有复杂年内波动（如双峰：寒假+暑假） |
| 周季节性 `weekly` | **3** | 用 3 对 sin/cos 拟合周内模式 | 一周内有细微的日间差异 |
| 日季节性 `daily` | **4** | 用 4 对 sin/cos 拟合日内模式 | 小时级数据有复杂日内波动 |

> **为什么周季节默认 N=3？** 一周只有 7 个点，3 对 sin/cos 已经足够描述"周一低、周三高、周五更高、周末回落"这种级别的高低变化。N 超过 7 会导致冗余——7 个数据点无法支撑 14 个参数。

### 4.1.4 季节性先验尺度 `seasonality_prior_scale`

和趋势变点一样，季节性也有先验（默认也是拉普拉斯先验的近似——实际上是正态先验，但在 stan 后端经过处理类似 L2 正则）：$$a_n, b_n \sim N(0, \sigma^2)$$

`seasonality_prior_scale` 就是这个 $\sigma$——控制季节性波动的"容许幅度"：

```python
# 全局季节性先验
model = Prophet(seasonality_prior_scale=10.0)  # 默认
```

| 值 | 效果 |
|-----|------|
| 0.01 | 季节性几乎被压平——"我不相信有季节波动" |
| 10.0 (默认) | 适度灵活，适合大多数场景 |
| 100+ | 季节性可以任意剧烈波动，容易过拟合 |

> **警惕**：默认 10.0 看似很大，但这和 y 的量纲有关。如果你的 y 值在百万级别，10.0 的波动很合理；如果 y 在 0-1 之间，需要调小。

---

## 4.2 三种内置季节性

Prophet 内置支持三种周期：

```python
model = Prophet(
    yearly_seasonality=True,    # 年周期（傅里叶阶数默认 10）
    weekly_seasonality=True,    # 周周期（傅里叶阶数默认 3）
    daily_seasonality=False,    # 日周期（默认关闭——日均以上数据不需要）
)
```

### 自动启停规则

Prophet 会根据数据的时间跨度自动判断：

| 数据跨度 | 自动启用的季节性 |
|----------|-----------------|
| ≥ 2 年 | `yearly_seasonality=True` |
| ≥ 2 周 且 ＜ 2 年 | `weekly_seasonality=True` |
| 亚日级数据（如每小时） | `daily_seasonality=True` |

> **手动显式指定可以覆盖自动判断**：`Prophet(yearly_seasonality=False)` 强制关闭。

### 指定阶数

```python
# 自定义傅里叶阶数
model = Prophet(
    yearly_seasonality=20,   # 用 20 对 sin/cos（更灵活的年度模式）
    weekly_seasonality=5,    # 用 5 对 sin/cos
)
```

---

## 4.3 自定义季节性

不是所有业务周期都是年/周/日。你可以添加任意周期。

### 4.3.1 使用内置方法

```python
# 月度季节性（周期 = 30.5 天，阶数 = 5）
model = Prophet()
model.add_seasonality(
    name='monthly',          # 自定义名称，会作为 forecast 的列
    period=30.5,             # 周期长度（天）
    fourier_order=5,         # 傅里叶阶数
    prior_scale=10.0,        # 该季节性的先验尺度（可覆盖全局）
)
```

### 4.3.2 常见自定义周期

| 业务场景 | 周期 (天) | 建议阶数 |
|---------|----------|----------|
| 月度模式 | `30.5` | 3-5 |
| 季度模式 | `91.25` | 5-10 |
| 双周模式（发薪日效应） | `14` | 3 |
| 半天模式（亚日级数据） | `0.5` | 4 |

### 4.3.3 高阶数 vs 低阶数的直观对比

```python
# 同一个月度周期，不同阶数的表现
model.add_seasonality(name='monthly_low', period=30.5, fourier_order=2)   # 粗略的月内起伏
model.add_seasonality(name='monthly_high', period=30.5, fourier_order=12) # 精细的月内模式
# 注：实际使用时不会同时加两个同名周期，这里仅为对比
```

---

## 4.4 加性 vs 乘性季节性

这是实际建模中**最常见、最容易被忽略**的一个设定。

### 4.4.1 直觉：波动幅度是否随趋势放大

```
加性季节性（默认）:                  乘性季节性:
y = trend + season                    y = trend × (1 + season)

  │    ╱╲   ╱╲                        │       ╱╲     ╱╲
  │   ╱  ╲ ╱  ╲                       │      ╱  ╲   ╱  ╲
  │  ╱    ╲    ╲╱                     │     ╱    ╲ ╱    ╲
  │ ╱      ╲                          │   ╱       ╲      ╲
  │╱        ╲___                      │ ╱          ╲      ╲___
  └────────────────                    └────────────────────────
  波动幅度恒定                          波动幅度随趋势变大
```

| 场景 | 季节性类型 | 特征 |
|------|-----------|------|
| 夏天比冬天每天多卖 100 杯奶茶（绝对量恒定） | **加性** | 季节性波动不随趋势放大 |
| 夏天销量是冬天的 1.5 倍（比例恒定） | **乘性** | 趋势增长时，绝对波动也放大 |
| 新用户注册（快速增长期）的周末效应 | **乘性** | 趋势涨了，周末效应也跟着涨 |

### 4.4.2 设置方式

```python
# 全局乘性季节性
model = Prophet(seasonality_mode='multiplicative')

# 单个季节性指定模式
model = Prophet()
model.add_seasonality(name='yearly', period=365.25, fourier_order=10, mode='additive')
model.add_seasonality(name='weekly', period=7, fourier_order=3, mode='multiplicative')
```

### 4.4.3 如何判断用哪种？

最简单的方法：**画图看**。

```python
import matplotlib.pyplot as plt
import numpy as np

# 观察原始序列，波动幅度是否随趋势放大
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df['ds'], df['y'])
ax.set_title('检查：波动幅度是否随整体水平放大？')

# 数值判断：用对数变换看
df['log_y'] = np.log(df['y'])
# 如果 log(y) 的波动幅度看起来更均匀 → 原始序列是乘性 → 用 multiplicative
```

> **经验法则**：如果序列值跨越了一个数量级以上（比如从 100 涨到 10,000），考虑乘性季节性。如果序列相对平稳（在 100-300 之间波动），加性就够了。

---

## 4.5 条件季节性

某些季节性只在特定条件下出现——比如"只有节假日季的周末才出现特殊模式"。Prophet 通过 `condition_name` 支持。

```python
# 节假日季的周末模式
df['is_holiday_season'] = df['ds'].apply(
    lambda x: 'yes' if x.month in [11, 12, 1] else 'no'
)
future['is_holiday_season'] = future['ds'].apply(
    lambda x: 'yes' if x.month in [11, 12, 1] else 'no'
)

model = Prophet(weekly_seasonality=False)
model.add_seasonality(
    name='weekly_during_holiday_season',
    period=7,
    fourier_order=3,
    condition_name='is_holiday_season',  # 只在 is_holiday_season='yes' 时激活
)
model.fit(df)
```

---

## 4.6 季节性建模的完整例子

```python
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# ============================================================
# 场景：电商日销售数据（有明显年周期 + 周周期 + 双11月效应）
# ============================================================
df = pd.read_csv('sales_data.csv')

model = Prophet(
    # 全局设置
    seasonality_mode='multiplicative',      # 销量随规模放大，用乘性
    seasonality_prior_scale=15.0,           # 调大一点点，允许更强的季节波动

    # 内置季节性
    yearly_seasonality=15,                  # 零售年模式复杂（双11、618等多峰）
    weekly_seasonality=5,                   # 精确刻画工作/周末差异
    daily_seasonality=False,
)

# 自定义：月度季节性（月初发薪效应、月末清仓）
model.add_seasonality(
    name='monthly',
    period=30.5,
    fourier_order=5,
    mode='multiplicative',
)

# 自定义：季度季节性
model.add_seasonality(
    name='quarterly',
    period=91.25,
    fourier_order=8,
    mode='multiplicative',
)

model.fit(df)
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

# 查看各季节性分量
seasonal_cols = ['yearly', 'weekly', 'monthly', 'quarterly']
print(forecast[['ds'] + seasonal_cols].tail(10))

# 成分分解图
model.plot_components(forecast)
plt.show()
```

---

## 4.7 常见问题速查

| 问题 | 原因 | 解决 |
|------|------|------|
| 年季节性看起来太像正弦波 | 傅里叶阶数太低 | 增大 `yearly_seasonality` 阶数 |
| 周季节性曲线剧烈抖动 | 阶数过高 + 数据不足 | 减小 `weekly_seasonality` 阶数 |
| 预测的季节性波动被放得很大 | 乘性季节性 + 趋势外推太远 | 缩短预测 horizon，或改用加性 |
| 自定义季节性加了但没看到效果 | `prior_scale` 太小，被先验压平 | 增大该季节性的 `prior_scale` |
| 季节性分量图出现奇怪的模式 | 条件季节性的 `condition_name` 列在未来数据中未设置 | 确保 `future` 中也有该列 |

---

## 4.8 核心概念清单

| 概念 | 一句话理解 |
|------|-----------|
| **傅里叶级数** | 用若干对 sin/cos 波叠加来逼近任意周期形状 |
| **傅里叶阶数 N** | sin/cos 的对数——决定季节性曲线能有多"曲折" |
| **加性季节性** | 波动幅度恒定，与趋势水平无关 |
| **乘性季节性** | 波动幅度随趋势水平同比缩放 |
| **seasonality_prior_scale** | 对季节性波动的"信任度"——小 = 压平，大 = 灵活 |
| **条件季节性** | 只在特定条件下激活的季节性模式 |

### 参数速查

| 参数 | 默认值 | 何时动 |
|------|--------|--------|
| `yearly_seasonality` | `'auto'` / `10` | 年内多峰 → 调大阶数；平稳 → 可关闭 |
| `weekly_seasonality` | `'auto'` / `3` | 日间差异精细 → 调到 5-7 |
| `daily_seasonality` | `'auto'` / `4` | 仅亚日级数据需要 |
| `seasonality_mode` | `'additive'` | 数值跨数量级 → `'multiplicative'` |
| `seasonality_prior_scale` | `10.0` | 季节性过弱 → 调大；过强 → 调小 |

---

下一章深入节假日与特殊事件建模——Prophet 区别于传统时序方法的核心优势之一。
