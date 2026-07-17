# 第 5 章：节假日与特殊事件

节假日建模是 Prophet 区别于传统时间序列方法（ARIMA、ETS 等）的**核心差异化能力**。传统方法需要你手动构造虚拟变量塞进外部回归项，Prophet 把这个过程内置了。

---

## 5.1 为什么节假日不能交给季节性

直觉上你会想："春节每年一次，这不就是年季节性的一部分吗？"

问题在于：

| 特征 | 普通季节性 | 节假日 |
|------|-----------|--------|
| **发生时间** | 固定日期（每年 12 月 25 日） | ✅ 固定 | ✅ 固定（春节除外） |
| **影响范围** | 精确到天，无前后溢出 | ✅ | ❌ 节假日前后也有影响 |
| **模式形状** | 平滑、连续（傅里叶级数） | ✅ | ❌ **脉冲式**——一天极高，次日回落 |
| **与其他效应叠加** | 自然叠加在趋势上 | ✅ | ❌ 节假日可能改变"周末效应"的含义 |

> **核心矛盾**：傅里叶级数适合刻画**平滑渐变**的周期模式，而节假日效应是**尖锐的、离散的脉冲**。用 sin/cos 去拟合一个单日脉冲需要极高的阶数，会导致其他日期被误伤。

Prophet 的解法：**将节假日作为独立的虚拟变量组件 $h(t)$，与趋势 $g(t)$、季节 $s(t)$ 并行建模。**

---

## 5.2 节假日模型的工作原理

### 5.2.1 数学形式

对于每个节假日 $i$，Prophet 在其发生日期 $t_i$ 及其前后窗口生成一个指示变量：

$$h(t) = \sum_{i} \kappa_i \cdot \mathbf{1}[t \in \text{window}_i]$$

- $\kappa_i$：节假日 $i$ 的效应强度（从数据中学习）
- $\text{window}_i$：节假日影响的时间窗口
- 先验：$\kappa_i \sim N(0, \nu^2)$，其中 $\nu$ = `holidays_prior_scale`（默认 10.0）

### 5.2.2 关键洞察：窗口的概念

节假日的影响几乎从不局限于当天：

```
               前窗口                   后窗口
              ┌──┴──┐               ┌──┴──┐
  ────────────●●●●●●●───────────────●●●●●●●────────────
             -2 -1  0              +1 +2 +3
              前夕  当天            节后回暖
              "买年货" "除夕"        "处理退货"

春节影响窗口: lower_window=-2, upper_window=+3
```

| 窗口参数 | 含义 | 典型场景 |
|----------|------|----------|
| `lower_window` | 节前影响天数（负数） | 春节前采购（-7）、黑五前预热（-3） |
| `upper_window` | 节后影响天数 | 春节后返工（+3）、黑五后处理退货（+2） |
| 窗口内每一天 | 该日也属于节假日影响范围 | 每个窗口日共享**同一个 $\kappa_i$** |

> ⚠️ 窗口内每一天使用**相同的效应强度 $\kappa_i$**。如果需要窗口内各天效应不同（比如除夕 vs 大年初一是完全不同的行为），需要把它们定义为**不同的节假日**。

---

## 5.3 代码实操

### 5.3.1 最简单用法：内置国家节假日

```python
from prophet import Prophet

model = Prophet()
model.add_country_holidays(country_name='CN')  # 中国节假日
model.fit(df)
```

Prophet 内置了多国节假日日历，底层数据来自 `holidays` 包：

| 常用国家代码 | 包含的节假日 |
|-------------|-------------|
| `'CN'` | 春节、清明、劳动节、端午、中秋、国庆等 |
| `'US'` | New Year, MLK Day, Memorial Day, July 4th, Labor Day, Thanksgiving, Christmas |
| `'BR'` | Carnaval, Tiradentes, etc. |

```python
# 查看所有支持的国家
from prophet.make_holidays import make_holidays_df
print(make_holidays_df.__doc__)
```

### 5.3.2 自定义节假日

```python
# ⚠️ 列名必须是 holiday 和 ds！这是初学者常掉的坑。
holidays = pd.DataFrame({
    'holiday': ['promo_618', 'promo_618', 'promo_1111', 'promo_1111',
                'company_anniversary'],
    'ds': pd.to_datetime(['2023-06-18', '2024-06-18',
                          '2023-11-11', '2024-11-11',
                          '2024-03-15']),
    # 可选：窗口（不写则仅当天）
    'lower_window': [-3, -3, -2, -2, -1],  # 618前3天开始预热
    'upper_window': [1, 1, 3, 3, 1],       # 双11后3天处理尾款/退换
})

model = Prophet(holidays=holidays)
model.fit(df)
```

### 5.3.3 完整示例：中国电商场景

```python
import pandas as pd
from prophet import Prophet

# ============================================================
# 场景：中国电商日销售数据
# 需要建模：内置中国节日 + 电商促销节 + 自定义窗口
# ============================================================

# 1. 定义电商专属大促日
ecommerce_events = pd.DataFrame({
    'holiday': [
        # 618 大促（6月1日~6月18日）
        '618_start',
        '618_peak',
        # 双11（11月1日~11月11日）
        '1111_preheat',
        '1111_peak',
        # 双12
        '1212_peak',
        # 公司周年庆
        'anniversary',
    ],
    'ds': pd.to_datetime([
        '2023-06-01',   # 618预热期开始
        '2023-06-18',   # 618当天峰值
        '2023-11-01',   # 双11预热
        '2023-11-11',   # 双11正日
        '2023-12-12',   # 双12
        '2024-03-15',   # 公司周年庆
    ]),
    'lower_window': [0, -2, 0, -3, -1, 0],
    'upper_window': [17, 3, 10, 5, 1, 1],
})
# 618_start: 从6月1日起影响17天（至6月18日）
# 1111_preheat: 从11月1日起影响10天（至11月11日）
# 1111_peak: 峰值从11月8日起影响5天（至11月16日）

# 2. 创建模型
model = Prophet(
    holidays=ecommerce_events,          # 自定义促销日
    holidays_prior_scale=15.0,          # 给促销日更大灵活性
    yearly_seasonality=15,
    weekly_seasonality=5,
    seasonality_mode='multiplicative',  # 促销效应是比例放大的
)

# 3. 叠加内置中国节日（自动识别重复日期）
model.add_country_holidays(country_name='CN')

model.fit(df)

# 4. 可视化节假日效应
forecast = model.predict(model.make_future_dataframe(periods=90))
print("节假日效应列:")
print(forecast[['ds', 'holidays']].dropna().head(20))

model.plot_components(forecast)
```

### 5.3.4 查看所有被识别的节假日

```python
# 模型拟合后查看完整的节假日表
model.train_holiday_names
# 输出示例:
# frozenset({'618_start', '618_peak', '1111_preheat', '1111_peak',
#            '1212_peak', 'anniversary',
#            '春节', '清明节', '劳动节', '端午节', '中秋节', '国庆节',
#            '元旦', '抗战胜利纪念日'})
```

---

## 5.4 节假日先验尺度 `holidays_prior_scale`

和变点、季节性的先验一样，节假日也有一把控制"松紧"的旋钮：

| 参数 | 默认值 | 效果 |
|------|--------|------|
| `holidays_prior_scale` | `10.0` | 全局——所有节假日的效应宽松度 |

```python
model = Prophet(holidays=my_holidays, holidays_prior_scale=5.0)  # 更保守
```

也可以在单条节假日上覆盖：

```python
# 在 DataFrame 加一列 prior_scale 来覆盖全局设置
holidays = pd.DataFrame({
    'holiday': ['big_event', 'small_event'],
    'ds': pd.to_datetime(['2023-12-25', '2023-03-01']),
    'prior_scale': [20.0, 2.0],  # 大事件允许大波动，小事件限制在窄范围
})
```

---

## 5.5 实战模式与经验

### 5.5.1 模式一：促销日叠加

当你既有内置节日又有自定义促销时，某些日期会重叠。Prophet 的处理方式是**各自独立的效应相加**——春节 + 情人节促销会叠加。

```python
# 不会冲突——Prophet 把它们当成独立的回归项
model.add_country_holidays(country_name='CN')       # 春节
model = Prophet(holidays=promo_events)              # 情人节促销
```

### 5.5.2 模式二：滚动节假日

农历节假日（春节、端午、中秋）每年日期不同。Prophet 内置的中国节假日已处理农历转换。自定义时需要注意：

```python
# ✅ 正确：每年独立一行
holidays = pd.DataFrame({
    'holiday': ['spring_festival'] * 3,
    'ds': pd.to_datetime(['2023-01-22', '2024-02-10', '2025-01-29']),
})

# ❌ 错误：仅定义一年——Prophet 不会自动推算其他年份
holidays = pd.DataFrame({
    'holiday': ['spring_festival'],
    'ds': pd.to_datetime(['2023-01-22']),
})
# 这会导致 2024 年的春节没有被建模！
```

> **规则**：自定义节假日的 `ds` 必须覆盖未来预测期内的所有发生日期。`make_future_dataframe` 只生成日期列，不会自动推算节假日。

### 5.5.3 模式三：周期性事件的"窗口"策略

有些事件虽然有周期，但用季节性建模太重、用节假日建模更精准：

| 业务事件 | 建模方式 | 理由 |
|----------|---------|------|
| 每月1号发薪日 | 自定义季节性（`period=30.5`） | 周期性规律，不适合逐月写 |
| 黑色星期五 | 节假日 + 窗口 | 每年日期不同（11月第四个周四），脉冲式 |
| 新产品发布日 | 节假日 | 不规律，手动指定日期 |
| 季度财报发布 | 节假日 | 日期不固定，且每次窗口长度可能不同 |

### 5.5.4 模式四：负效应节假日

不是所有节假日都让指标上涨。某些节日可能导致业务指标**下降**——比如 B2B 业务在春节期间的停滞。

Prophet 自动处理：$\kappa_i$ 可以是负值。不需要特殊配置，模型从数据中学习效应方向。

---

## 5.6 节假日与乘性季节性的交互

当 `seasonality_mode='multiplicative'` 时，节假日效应也变为乘性：

```python
# 乘性下：促销日销量 = 趋势 × (1 + 季节性) × (1 + 节假日效应)
model = Prophet(
    holidays=promo_events,
    seasonality_mode='multiplicative',
)
```

这意味着同样是大促，在业务体量变大后，促销的**绝对增量**也会放大——这个行为在大多数零售场景中是合理的。

```python
# 如果你需要某些节假日是乘性、其余是加性，目前 Prophet 不支持单日指定
# 变通方案：用额外回归量（第 6 章）手动构造乘性效应
```

---

## 5.7 常见问题速查

| 问题 | 原因 | 解决 |
|------|------|------|
| 内置中国节日不生效 | 地区代码写错 | 必须是 `'CN'`，不是 `'cn'` 或 `'China'` |
| 自定义节假日没出现 | 列名写错 | 必须 `holiday` 和 `ds`，不是 `name` 和 `date` |
| 节假日效应对未来预测为 0 | 未来日期没有在 `holidays` DataFrame 中 | 确保节假日覆盖了未来预测期 |
| 多个节假日重叠产生异常值 | 效应简单相加，可能过度放大 | 减少重叠日的节假日定义，或用额外回归量手动控制 |
| 节假日效应看起来太小 | `holidays_prior_scale` 过紧 | 调大 `holidays_prior_scale` |
| 节假日效应过于剧烈 | 先验太松，过拟合了少量节假日样本 | 调小 `holidays_prior_scale` |

---

## 5.8 核心概念清单

| 概念 | 一句话理解 |
|------|-----------|
| **节假日组件 $h(t)$** | 与趋势、季节并行的第三大 GAM 组件，专门建模脉冲式效应 |
| **窗口 (lower/upper window)** | 节假日影响的前/后溢出天数，共享同一个效应强度 |
| **holidays_prior_scale** | 节假日效应的"信任旋钮"——小 = 保守估计节假日影响 |
| **内置国家日历** | Prophet 自带的多国法定节假日，底层数据来自 Python `holidays` 包 |
| **负效应** | 节假日也可以降低指标值，模型自动学习方向 |

### 参数速查

| 参数 / 方式 | 作用 | 注意事项 |
|-------------|------|----------|
| `holidays=DataFrame` | 传入自定义节假日 | 列名必须 `holiday` + `ds`，窗口列名 `lower_window` / `upper_window` |
| `add_country_holidays('CN')` | 加载内置国家节假日 | 大小写敏感 |
| `holidays_prior_scale=10.0` | 全局节假日先验 | 可按单日覆盖（在 DataFrame 中加 `prior_scale` 列） |
| `seasonality_mode` | 节假日效应是加性还是乘性 | 与全局 `seasonality_mode` 保持一致 |

---

下一章讲额外回归量——把你自己的外部变量（价格、天气、营销投入）塞进 Prophet 模型。
