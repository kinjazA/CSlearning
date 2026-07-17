# 总结：Prophet 核心回顾与进阶扩展

> 快速复盘手册——涵盖核心公式、参数速查、决策框架、以及 Prophet + GBDT 混合预测。

---

## 一、Prophet 的本质：一句话

**Prophet = 贝叶斯 GAM，专为业务时间序列设计。**

你把时间序列拆成四块——趋势、季节、节假日、外部变量——每块独立建模，最后加在一起。每个块有一个"松紧旋钮"（先验尺度），让你在"相信数据"和"防止过拟合"之间调节。

$$y(t) = g(t) + s(t) + h(t) + r(t) + \varepsilon$$

---

## 二、参数速查表

### 你最常调的参数（按频率排序）

| 参数 | 默认 | 作用 | 调大 → | 调小 → |
|------|------|------|--------|--------|
| `changepoint_prior_scale` | `0.05` | 趋势灵活度 | 趋势更能弯（追噪声风险） | 趋势更直（欠拟合风险） |
| `seasonality_prior_scale` | `10.0` | 季节性灵活度 | 季节波动更大 | 季节更平 |
| `seasonality_mode` | `'additive'` | 季节叠加方式 | `'multiplicative'`—波动按比例 | — |
| `holidays_prior_scale` | `10.0` | 节假日效应幅度 | 节假日影响更大 | 节假日影响被压缩 |
| `yearly_seasonality` | `auto` / `10` | 年季节傅里叶阶数 | 年内模式更复杂 | 年内模式更圆滑 |
| `weekly_seasonality` | `auto` / `3` | 周季节傅里叶阶数 | 周内更精细 | 周内更平滑 |
| `interval_width` | `0.80` | 预测区间宽度 | 更宽（更保守） | 更窄 |
| `uncertainty_samples` | `1000` | 区间平滑度 | 更平滑（更慢） | 更快（更粗糙）；`0`=关闭 |

### 偶尔调的参数

| 参数 | 默认 | 何时动 |
|------|------|--------|
| `growth` | `'linear'` | 有明确的物理上限 → `'logistic'` |
| `n_changepoints` | `25` | 数据特别短 → 减少；特别长 → 可增加 |
| `changepoint_range` | `0.8` | 想让最近数据也能触发变点 → 调大到 0.9-0.95 |
| `mcmc_samples` | `0` | 需要完整后验分布 → 设为 300+ |

---

## 三、决策速查

### 这个参数该调吗？

```
数据有明显的趋势变化吗？
  ├── 是 → 调 changepoint_prior_scale
  └── 否 → 保持默认 0.05

季节性波动幅度是否随业务体量放大？
  ├── 是（体量翻了 3 倍以上）→ seasonality_mode='multiplicative'
  └── 否 → 保持 additive

残差的周内模式是否显著不为零？
  ├── 是 → 增大 weekly_seasonality 阶数或 seasonality_prior_scale
  └── 否 → 保持默认

预测区间是否太窄（实际值经常跑出去）？
  ├── 是 → 开启 MCMC 或调大 changepoint_prior_scale
  └── 否 → 保持默认

数据是否有节假日/促销日效应？
  ├── 是 → 自定义 holidays DataFrame
  └── 否 → 不需要

是否有可用的外部变量（价格、天气、广告）？
  ├── 是 → 用 add_regressor，并解决未来值问题
  └── 否 → 不需要
```

---

## 四、全流程速查

```
1. 数据质量检查
   └── data_quality_report(df)  →  缺失？异常值？跨度？季节性？

2. 基线模型
   └── Prophet() 默认参数 → 评估 RMSE/残差 → 发现问题

3. 粗调
   └── 选 mode → 调 cps × sps 组合 → 热力图找最优区域

4. 细调
   └── 节假日先验 → 傅里叶阶数 → 回归量先验

5. 交叉验证
   └── cross_validation → performance_metrics → 残差分析 → 区间覆盖率

6. 部署
   └── 序列化 → 管道 → 监控 → 自动重训练
```

---

## 五、常见错误与规避

| 错误 | 正确做法 |
|------|----------|
| 默认参数跑一遍就下结论"Prophet 不行" | 至少调 `changepoint_prior_scale` 和 `seasonality_prior_scale` |
| 在全部数据上调参，然后把调出来的 RMSE 当作预期精度 | 保留一份未触碰的测试集（最新数据）做最终验证 |
| 加性模型用在跨数量级的序列上 | 体量翻 3 倍以上 → 试乘性，或对 y 取 log |
| 傅里叶阶数盲目调大 | 周季节 N > 7 无意义（一周就 7 个点） |
| 额外回归量忘了填充未来值 | `future['regressor'] = ...` ——整个流程最容易漏的一行 |
| 异常值不分青红皂白全删了 | "明天还会再发生吗？" → 是=保留建模，否=NA 化 |
| 预测 horizon 超过历史长度的 1/2 | 趋势不确定性会大到预测失去参考价值 |

---

## 六、Prophet + GBDT：把 Prophet 当特征提取器

### 6.1 这是什么思路

Prophet 的核心能力是**分解**——把一条时间序列拆成趋势、季节、节假日。这些分解后的分量本身就是高质量的特征。

把这些分量**喂给梯度提升树模型（XGBoost / LightGBM / CatBoost）**，让树模型在 Prophet 的基础上学习更复杂的非线性交互——这就是 Prophet + GBDT 混合预测。

```
传统方式:
  原始数据 → Prophet → 预测值 yhat

混合方式:
  原始数据 → Prophet → 分量(trend, yearly, weekly, holidays, ...)
                         │
                         ├──→ 作为特征 ──→ GBDT → 最终预测
                         │
  外部特征(价格、天气、广告等) ──→ 也喂给 GBDT ──┘
```

### 6.2 为什么有效

| Prophet 擅长的 | GBDT 擅长的 | 混合后的效果 |
|---------------|------------|------------|
| 捕捉趋势和季节结构 | 处理大量异构特征 | Prophet 提取时间结构，GBDT 做"最后一公里" |
| 平滑外推未来模式 | 非线性交互（如"周末 + 促销 + 雨天"的三阶交互） | GBDT 学习 Prophet 没捕获的残差信号 |
| 可解释的分解 | 自动特征选择 + 缺失值处理 | Prophet 的分量本身可解释，GBDT 补充精度 |

这不是"两种模型取平均"的简单集成——这是一种**层次化建模 (Hierarchical Modeling)**：下层 Model 学习数据生成的结构，上层 Model 学习具体的特征交互。

### 6.3 代码实现

```python
import pandas as pd
import numpy as np
from prophet import Prophet
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

# ============================================================
# Step 1: Prophet 分解 → 提取分量作为特征
# ============================================================
model_prophet = Prophet(
    yearly_seasonality=15,
    weekly_seasonality=5,
    changepoint_prior_scale=0.1,
)
model_prophet.fit(df_train)

# 对训练集和测试集的所有日期生成 Prophet 分量
all_dates = pd.concat([df_train[['ds']], df_test[['ds']]])
prophet_components = model_prophet.predict(all_dates)

# 提取分量为特征
feature_cols = ['trend', 'yearly', 'weekly', 'holidays']
# 如果有额外回归量，也加进去
extra_cols = [c for c in prophet_components.columns
              if c not in ('ds', 'yhat', 'yhat_lower', 'yhat_upper',
                           'trend_lower', 'trend_upper',
                           'additive_terms', 'multiplicative_terms')]
feature_cols.extend([c for c in extra_cols if c in prophet_components.columns])

df_features = prophet_components[['ds'] + feature_cols].copy()
# ⚠️ 注意：不能把 yhat 作为特征——那是 Prophet 的最终预测，会造成数据泄漏
# 只用 trend / yearly / weekly / holidays 等分解分量

# ============================================================
# Step 2: 加入其他外部特征
# ============================================================
# 假设你有这些额外特征
df_features = df_features.merge(df_external_features, on='ds', how='left')

# 时间特征（GBDT 本身不感知时间结构）
df_features['dayofweek'] = df_features['ds'].dt.dayofweek
df_features['month'] = df_features['ds'].dt.month
df_features['dayofyear'] = df_features['ds'].dt.dayofyear
df_features['is_weekend'] = (df_features['dayofweek'] >= 5).astype(int)

# ============================================================
# Step 3: 构建训练集（注意：不能用随机 CV！）
# ============================================================
train_features = df_features[df_features['ds'].isin(df_train['ds'])]
test_features = df_features[df_features['ds'].isin(df_test['ds'])]

target_col = 'y'
X_train = train_features.drop(columns=['ds'])
y_train = df_train[target_col]
X_test = test_features.drop(columns=['ds'])
y_test = df_test[target_col]

# ============================================================
# Step 4: 训练 GBDT
# ============================================================
model_gbdt = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
)

# ⚠️ 时序交叉验证——不能用 shuffle！
tscv = TimeSeriesSplit(n_splits=5)

# 在时间序列 CV 上评估
cv_scores = []
for train_idx, val_idx in tscv.split(X_train):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model_gbdt.fit(X_tr, y_tr,
                   eval_set=[(X_val, y_val)],
                   callbacks=[lgb.early_stopping(50)])
    pred = model_gbdt.predict(X_val)
    cv_scores.append(mean_absolute_error(y_val, pred))

print(f"GBDT TimeSeries CV MAE: {np.mean(cv_scores):.1f} ± {np.std(cv_scores):.1f}")

# 最终在全部训练集上拟合
model_gbdt.fit(X_train, y_train)

# ============================================================
# Step 5: 对比：纯 Prophet vs 混合模型
# ============================================================
# 纯 Prophet
prophet_pred = model_prophet.predict(df_test[['ds']])['yhat']
mae_prophet = mean_absolute_error(y_test, prophet_pred)

# 混合模型
gbdt_pred = model_gbdt.predict(X_test)
mae_hybrid = mean_absolute_error(y_test, gbdt_pred)

print(f"\n纯 Prophet MAE:     {mae_prophet:.1f}")
print(f"Prophet + GBDT MAE: {mae_hybrid:.1f}")
print(f"提升:                {(1 - mae_hybrid/mae_prophet)*100:.1f}%")

# ============================================================
# Step 6: 特征重要性分析
# ============================================================
importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model_gbdt.feature_importances_,
}).sort_values('importance', ascending=False)

print("\n特征重要性 Top 10:")
print(importance.head(10).to_string(index=False))
```

### 6.4 为什么不能把 `yhat` 当特征

这是混合模型最常见的错误：

```python
# ❌ 错误——数据泄漏
df_features['prophet_prediction'] = prophet_components['yhat']
# yhat 已经是 Prophet 对 y 的最终预测，
# GBDT 看到它后几乎不需要学别的——这等于用 Prophet 作弊

# ✅ 正确——只用分解分量
df_features['trend'] = prophet_components['trend']
df_features['yearly'] = prophet_components['yearly']
df_features['weekly'] = prophet_components['weekly']
# 这些分量不包含"最终答案"，GBDT 需要自己学会组合它们
```

### 6.5 这个模式是广泛使用的方法吗

**是的。** 它有几个名字，但思路相同：

| 名称 | 出处/场景 |
|------|----------|
| **Prophet + XGBoost hybrid** | Kaggle M5 等时序竞赛中的常用方案 |
| **Forecast stacking / blending** | 将 Prophet 预测作为元特征加入元模型 |
| **Feature-based forecasting** | 将时间序列分解为特征矩阵，再用任何 ML 模型 |
| **Decomposition + ML** | 学术文献中的正式名称 |

为什么被广泛采用：

1. **取长补短**——Prophet 懂时间结构（趋势、周期、节假日），GBDT 懂特征交互（"周末 + 促销 + 下雨"三阶交叉效应）。单独用哪个都不完美。
2. **工程友好**——不需要修改 Prophet 或 GBDT 的源码，只是把两个现成工具串起来。
3. **可解释性不丢**——Prophet 的分量图仍然可以给业务方看（"你看，趋势在涨、周末效应是 +20%——GBDT 只是在这基础上做微调"）。
4. **竞赛验证**——在 M5、Walmart Recruiting 等时序竞赛中，Prophet 分量 + LightGBM 是榜单前列的常见组件。

### 6.6 什么时候该用混合模型

| 场景 | 建议 |
|------|------|
| 数据有明显的趋势+季节+节假日 | ✅ Prophet 分量很有价值 |
| 同时有 5+ 个外部特征 | ✅ GBDT 处理多特征的交互比 Prophet 强 |
| 时序组件和外部特征之间有交互（如"周末 + 促销"） | ✅ GBDT 天然学习交互 |
| 只有一条序列且没有外部特征 | ❌ 纯 Prophet 就够了，加 GBDT 是过度设计 |
| 对预测延迟有严格要求（毫秒级） | ⚠️ 两个模型串行会增加延迟 |
| 必须 100% 可解释（如监管场景） | ⚠️ GBDT 的黑箱性会削弱整体可解释性 |

### 6.7 进阶：不只是 Prophet

这个"分解 + ML"的思路不限于 Prophet。任何能拆解时间序列的方法都可以：

```
时间序列分解源:
  ├── Prophet → trend, yearly, weekly, holidays
  ├── STL分解 → trend, seasonal, residual
  ├── MSTL分解 → 多季节分解
  ├── 移动平均 → 趋势分量
  └── 傅里叶变换 → 频域特征

         ↓  全部作为特征喂给  ↓

ML 模型:
  ├── LightGBM / XGBoost / CatBoost
  ├── Random Forest
  └── 甚至简单线性回归（如果特征工程做得够好）
```

---

## 七、核心概念总览

```
                     Prophet 知识地图
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
       数据层             模型层            工程层
          │                │                │
    ┌─────┴─────┐    ┌─────┴─────┐    ┌─────┴─────┐
    │ 缺失容忍   │    │ 趋势 g(t) │    │ 交叉验证   │
    │ 异常检测   │    │ 季节 s(t) │    │ 超参调优   │
    │ 粒度选择   │    │ 节假日h(t)│    │ 持久化     │
    │ 质量清单   │    │ 回归量r(t)│    │ 监控告警   │
    └───────────┘    │ 不确定性  │    │ 大规模预测 │
                     └─────┬─────┘    └───────────┘
                           │
                     ┌─────┴─────┐
                     │ GAM 框架  │
                     │ 贝叶斯推断 │
                     │ 先验→后验 │
                     └───────────┘
                           │
                     ┌─────┴─────┐
                     │ 混合预测  │
                     │ + GBDT   │
                     └───────────┘
```

---

## 八、学习检查清单

在认为自己"掌握了 Prophet"之前，逐一确认：

- [ ] 能解释 GAM 三个字母分别代表什么，以及为什么 Prophet 的 "G" 不太准确
- [ ] 能解释拉普拉斯先验为什么让变点稀疏，以及 `changepoint_prior_scale` 如何控制稀疏度
- [ ] 能解释傅里叶阶数如何影响季节性曲线的灵活性
- [ ] 能区分"加性"和"乘性"季节性，知道什么数据特征触发切换
- [ ] 能设计节假日 DataFrame（含窗口），理解内置国家节假日的使用和限制
- [ ] 能使用额外回归量，并清楚未来值填充是使用者的责任
- [ ] 能解释 MAP 和 MCMC 的区别，知道何时开启 MCMC
- [ ] 能用 `cross_validation` + `performance_metrics` 做完整的模型评估
- [ ] 能读懂残差图——区分系统偏差、欠拟合、过拟合
- [ ] 能执行分阶段超参数调优（mode → cps×sps → 细调）
- [ ] 能在建模前跑数据质量检查清单
- [ ] 能将模型序列化为 JSON 并成功加载
- [ ] 能实现 Prophet + LightGBM 的混合预测，理解为什么只用分量不用 yhat
- [ ] 能用自己的业务数据跑完"数据→Prophet→CV→调参→部署"全流程

---

> 回到 [课程总纲](./README.md) 查看完整 13 章目录。
