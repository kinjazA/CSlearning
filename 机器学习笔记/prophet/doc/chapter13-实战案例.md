# 第 13 章：实战案例

前面 12 章是工具箱，这一章是**怎么组合使用这些工具**。四个案例覆盖最常见的业务场景，每个都遵循"分析 → 建模 → 评估 → 部署"的完整流程。

---

## 案例一：零售日销售预测

### 场景

某连锁超市，预测未来 90 天每日销售额，用于采购计划和人员排班。

### 数据特征

- 3 年日销售数据（2022-2024）
- 明显的周季节性（周末高）、年季节性（春节/暑期高峰）
- 多次大促活动（618、双11、周年庆）
- 部分门店周日关门（y=0）

### 建模策略

```python
# 1. 业务判断 → 乘性季节性（体量逐年增长，季节性波动随之放大）
# 2. 傅里叶阶数 → 年=15（多峰形态），周=5（精确建模周末）
# 3. 节假日 → 自定义大促日 + 内置中国节日
# 4. 变点 → 给促销季前后手动加变点
# 5. 数据预处理 → 周日关门的 0 值是真实信号，保留

model = Prophet(
    seasonality_mode='multiplicative',
    yearly_seasonality=15,
    weekly_seasonality=5,
    changepoint_prior_scale=0.1,
    holidays_prior_scale=15.0,
    holidays=promo_events,
)
model.add_country_holidays(country_name='CN')
```

### 关键决策点

| 决策 | 选择 | 理由 |
|------|------|------|
| 加性/乘性 | 乘性 | 体量增长 3 倍，周末效应按比例放大 |
| 周日 0 值 | 保留 | 真实关门，不是数据错误 |
| 大促日 | 自定义节假日 + 3 天窗口 | 618/双11 效应在前后几天也有 |
| CV horizon | 90 天 | 匹配业务决策周期 |

### 评估结果

| 指标 | 值 | 解读 |
|------|-----|------|
| RMSE | ~850（日均销售额 ~15,000） | 误差约 5.7%，良好 |
| 区间覆盖率 | 82% | 接近 80% 期望，区间校准良好 |
| 预测衰减 | 第 90 天 RMSE 为第 7 天的 1.8 倍 | 长期预测仍可接受 |

---

## 案例二：网站日流量预测

### 场景

某 SaaS 官网，预测未来 30 天每日 UV，用于服务器弹性扩容。

### 数据特征

- 2 年日 UV 数据
- 明显的工作日/周末差异（工作日高）
- 节假日期间流量骤降
- 有几个"爆款文章日"导致的流量尖峰

### 建模策略

```python
# 1. 加性季节性（UV 波动幅度与总 UV 不成比例放大）
# 2. 周季节性阶数 = 7（工作日 vs 周末是核心特征）
# 3. 节假日 = 负效应（流量下降）
# 4. 爆款日 = 单独标记，用异常值处理（NA 化）

model = Prophet(
    seasonality_mode='additive',
    weekly_seasonality=7,
    yearly_seasonality=10,
    changepoint_prior_scale=0.1,
    holidays_prior_scale=20.0,   # 节假日效应可能很大
)
```

### 特殊处理

```python
# 爆款文章日 → 标记为 NA（不是常态）
viral_dates = ['2023-03-15', '2023-08-22', '2024-01-10']
df.loc[df['ds'].isin(viral_dates), 'y'] = None

# 节假日作为"负事件"
holidays = pd.DataFrame({
    'holiday': ['国庆假期', '春节假期', '五一假期'],
    'ds': pd.to_datetime(['2023-10-01', '2024-02-10', '2023-05-01']),
    'lower_window': 0,
    'upper_window': [6, 6, 4],  # 假期长度
})
```

### 关键决策点

| 决策 | 选择 | 理由 |
|------|------|------|
| 加性/乘性 | 加性 | UV 基数变化不大（2 倍以内） |
| 节假日 | 负效应 | 人们放假时不访问 SaaS 官网 |
| 爆款日 | NA 化 | 单次事件，不可复现，不应影响趋势 |
| 预测 horizon | 30 天 | 扩容决策以月为单位 |

---

## 案例三：餐厅日营收预测

### 场景

连锁餐厅，预测未来 60 天各门店日营收，用于食材采购和人员排班。

### 数据特征

- 4 年日营收数据（500+ 家门店）
- 外部因素影响大（天气、广告投放、周边活动）
- 部分门店为新开门店（数据量不足 6 个月）
- 节假日营收暴涨

### 建模策略

```python
# 1. 大规模预测（500+ 门店 → 多进程并行）
# 2. 额外回归量：温度、降雨、广告投放
# 3. 新店：减少季节性阶数（数据不够）
# 4. 老店：完整建模

def build_model_for_store(store_age_days):
    """根据门店年龄返回不同参数"""
    if store_age_days < 180:
        return Prophet(
            weekly_seasonality=3,       # 降阶
            yearly_seasonality=False,   # 关年季节（数据不够）
            changepoint_prior_scale=0.5,
        )
    elif store_age_days < 365:
        return Prophet(
            weekly_seasonality=5,
            yearly_seasonality=5,       # 降阶
            changepoint_prior_scale=0.1,
        )
    else:
        return Prophet(
            weekly_seasonality=5,
            yearly_seasonality=15,
            changepoint_prior_scale=0.05,
        )
```

### 外部回归量处理

```python
# 温度 — 非线性
df['temp_deviation'] = np.abs(df['temperature'] - 25)
model.add_regressor('temperature')
model.add_regressor('temp_deviation')

# 降雨 — 二值
model.add_regressor('is_rainy')

# 广告投放 — 滞后效应
df['ad_spend_lag1'] = df['ad_spend'].shift(1)
df['ad_spend_lag7'] = df['ad_spend'].shift(7)
model.add_regressor('ad_spend')
model.add_regressor('ad_spend_lag1')
model.add_regressor('ad_spend_lag7')
```

### 关键决策点

| 决策 | 选择 | 理由 |
|------|------|------|
| 并行策略 | 多进程 | 500+ 门店，串行太慢 |
| 新店 vs 老店 | 不同参数模板 | 新店数据不够，不能套用老店参数 |
| 温度 | 非线性特征 | U 形效应（太冷/太热都影响客流） |
| 广告投放 | 滞后项 | 广告效果有延迟 |
| 失败策略 | 记录 + 人工介入 | 新店可能预测失败，不阻塞管道 |

---

## 案例四：能源日负荷预测

### 场景

某区域电网，预测未来 14 天每日用电负荷，用于发电调度。

### 数据特征

- 5 年日用电量数据（含小时级拆分）
- 强烈的季节模式（夏天空调、冬天取暖）
- 温度是最重要的外部因素
- 工作日/周末模式差异明显
- 精度要求极高（调度失误代价大）

### 建模策略

```python
# 1. 乘性季节性（用电基数随经济增长逐年递增）
# 2. 温度 = 核心回归量（非线性：加热度日 HDD + 制冷度日 CDD）
# 3. MCMC 开启（需要准确的预测区间用于风险评估）
# 4. 短 horizon（14 天，精度衰减可控）

# 温度特征工程
df['hdd'] = np.maximum(18 - df['temperature'], 0)  # 取暖需求
df['cdd'] = np.maximum(df['temperature'] - 22, 0)  # 制冷需求

model = Prophet(
    seasonality_mode='multiplicative',
    yearly_seasonality=20,    # 高阶数 — 年度用电模式复杂
    weekly_seasonality=5,
    changepoint_prior_scale=0.01,  # 紧先验 — 用电结构变化慢
    seasonality_prior_scale=15.0,
    interval_width=0.90,
)

model.add_regressor('hdd')
model.add_regressor('cdd')

# 如果精度要求极高，开启 MCMC
# model = Prophet(..., mcmc_samples=500)
```

### 关键决策点

| 决策 | 选择 | 理由 |
|------|------|------|
| 温度处理 | HDD/CDD | 标准能源行业做法，U 形 1→ 分段线性 |
| horizon | 14 天 | 电力调度以周/双周为单位 |
| MCMC | 建议开启 | 调度决策需要准确的概率区间 |
| 趋势先验 | 紧（0.01） | 电力结构变化很慢 |
| 不确定性区间 | 90% | 电力行业风险管理标准 |

### 评估结果

| 指标 | 值 | 解读 |
|------|-----|------|
| MAPE | 2.8% | 极高精度——温度解释了大部分方差 |
| RMSE | ~1200 MWh（日均 ~50,000 MWh） | 2.4% |
| 区间覆盖率 | 88% | 接近 90% 期望 |

---

## 四案例决策对比总结

| 维度 | 零售 | 网站流量 | 餐厅 | 能源 |
|------|------|---------|------|------|
| **seasonality_mode** | multiplicative | additive | multiplicative | multiplicative |
| **年季节阶数** | 15 | 10 | 15（老店） | 20 |
| **预测 horizon** | 90 天 | 30 天 | 60 天 | 14 天 |
| **ch.changepoint_prior_scale** | 0.1 | 0.1 | 0.05-0.5 | 0.01 |
| **额外回归量** | 促销标记 | 无 | 温度/广告 | HDD/CDD |
| **MCMC** | 不需要 | 不需要 | 不需要 | 建议开启 |
| **并行** | 按门店 | 单序列 | 按门店 | 按区域 |
| **异常值策略** | 保留大促日 | NA 化爆款日 | NA 化数据错误 | 保留极端天气日 |

---

## 核心概念清单

| 概念 | 一句话理解 |
|------|-----------|
| **案例 ≠ 模板** | 每个案例的选择都是数据特征和业务需求驱动的——不是"这个参数好" |
| **Horizon 决策** | 预测 horizon = 业务决策周期，不是越长越好 |
| **MCMC 的触发条件** | 需要精确概率区间时开启——如调度、风控 |
| **异常值的"保留 vs 删除"** | 取决于"如果明天再发生一次，我希望模型看到它吗" |
| **大规模 ≠ 简单N倍** | 新店要降参数、失败要容错、监控要到位 |

---

> 恭喜你完成了全部 13 章的学习。回到 [总纲](./README.md) 可以总览全课程。
