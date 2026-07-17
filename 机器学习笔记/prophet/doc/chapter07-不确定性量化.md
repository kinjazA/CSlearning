# 第 7 章：不确定性量化

前几章我们一直在构建点预测 $\hat{y}$——"明天卖多少"。但业务决策不止需要点估计。库存备货要问"最坏情况会卖多少"，预算审批要问"这个预测有多靠谱"。**不确定性量化就是把"我猜大概是 1000"升级为"80% 概率在 850 到 1150 之间"。**

---

## 7.1 不确定性的三个来源

Prophet 的预测区间来自对以下三个不确定性来源的综合：

```
预测不确定性
    │
    ├── 趋势不确定性 ── 未来趋势方向会变吗？
    │    └── 来源：changepoint 的 δ 估计不精确
    │
    ├── 季节性不确定性 ── 季节模式的估计有多准？
    │    └── 来源：傅里叶系数的后验分布宽度
    │
    └── 观测噪声 ── 即使模型完美，数据本身有随机波动
         └── 来源：ε ～ N(0, σ²) 中的 σ
```

> 注意：Prophet 的预测区间**不包含**节假日效应和额外回归量的不确定性。这些被视为已知输入，不是模型要估计的随机变量。

---

## 7.2 默认模式：MAP 估计 + 趋势不确定性模拟

### 7.2.1 Prophet 默认做什么

Prophet 默认分两步给出预测区间：

**Step 1 — MAP 估计**：找到"最可能"的一组参数（最大后验估计），得到点预测 $\hat{y}$。

**Step 2 — 趋势模拟**：保持季节性和噪声参数不变，仅对**趋势组件**做蒙特卡洛采样——模拟趋势未来可能的走向——从而得到 `yhat_lower` 和 `yhat_upper`。

```python
# 这就是控制模拟精度的参数
model = Prophet(
    uncertainty_samples=1000,   # 模拟 1000 条趋势路径（默认）
    interval_width=0.80,        # 给出 80% 预测区间（默认）
)
```

### 7.2.2 趋势模拟的机制——Prophet 如何"随机生成 1000 条路径"

这是整个不确定性量化的核心，我们把它拆开看。

---

#### 历史教会了 Prophet 什么

回顾第 3 章：Prophet 在历史数据上找到了一些变点，每个变点有一个斜率变化量 $\delta_j$。

```
历史数据上的变点（示意）:

        δ₁=+0.3        δ₃=-0.1
  ──────┐  ╱╲         ╱
         │╱  ╲       ╱
         │    ╲     ╱
 δ₀≈0   │     ╲   ╱   δ₂=+0.15
         │      ╲ ╱
         │       ╲╱
         │
   fit 完成后，Prophet 知道:
   - 历史变点平均频率: 约每 N 天一个
   - 历史 δ 的分布: 均值为 0，但有几个明显偏大的
   - τ (changepoint_prior_scale) 控制了 δ 的"典型大小"
```

从这些历史 δ 中，Prophet 推断出：**未来的趋势变化，幅度大概和历史上差不多。**

---

#### 生成一条路径的完整过程

Prophet 在预测期内**均匀放置新的候选变点**（就像它在历史数据的前 80% 所做的那样），然后为每个候选变点**随机抽取一个 $\delta$**：

```
Step 1: 在预测期内均匀放候选变点

  历史               未来
  |········|    | · · · · · · · · |
                 ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑   候选变点位置（均匀分布）

Step 2: 对每个候选变点，从拉普拉斯分布中随机抽一个 δ

  δ ～ Laplace(0, τ)

  Laplace(0, τ) 长这样:
       ╱╲
      ╱  ╲
     ╱    ╲
  ──╱──────╲──
    -τ  0  +τ

  大多数抽样结果 ≈ 0（尖峰在 0）
  少数抽样结果 > τ 或 < -τ（厚尾）

Step 3: 用这些 δ 构造未来的趋势线

  基斜率 k ────────────────────────────
                │      ╱  (δ=+0.2 被抽中)
                │    ╱
                │  ╱        ╲  (δ=-0.1 被抽中)
                │╱            ╲
                │               ╲──────
                ↑        ↑        ↑
             变点1    变点2    变点3
            (δ≈0)                    (δ≈0)
```

---

#### 为什么每次生成的路径不一样

**因为每次抽 δ 是随机的。** 用伪代码来理解：

```python
import numpy as np

# 从历史数据中学习到的 τ
tau = 0.05  # changepoint_prior_scale

# 一条趋势路径的生成
def generate_one_trend_path(base_slope, n_future_days, n_changepoints, tau):
    # 在预测期内均匀放置变点
    changepoint_days = np.linspace(0, n_future_days, n_changepoints)

    # 对每个变点，从 Laplace(0, τ) 中随机抽一个 δ
    deltas = np.random.laplace(loc=0, scale=tau, size=n_changepoints)

    # 构造趋势线
    trend = []
    slope = base_slope
    for day in range(n_future_days):
        if day in changepoint_days:
            idx = changepoint_days.index(day)
            slope += deltas[idx]  # 在当前变点处调整斜率
        trend.append(slope * day)

    return trend

# 调用 5 次 → 5 条不同的路径
for i in range(5):
    path = generate_one_trend_path(base_slope=0.1, n_future_days=365, n_changepoints=25, tau=0.05)
    # 每次调用 np.random.laplace 返回不同的随机数 → 路径不同

# ⚠️ 一个常见的误解：
# Prophet 不是从"历史的具体 δ 值"里复抽（resample）
# 而是从 Laplace(0, τ) 这个分布里生成全新的随机数
# 
# 历史的作用只有一个：告诉你 τ 是多少
# 比如历史 δ 大多在 ±0.05 以内 → τ ≈ 0.05
# 然后模拟时：np.random.laplace(0, 0.05) → 每次全新抽取
#
# 这也解释了为什么预测区间一定比历史波动范围更宽：
# 未来可能抽到你历史上从未见过的 δ 值——厚尾保证了这一点
```

```python
# 实际运行这段代码看看差异
np.random.seed(42)
for i in range(5):
    deltas = np.random.laplace(0, 0.05, 25)
    nonzero = (np.abs(deltas) > 0.01).sum()
    print(f"路径 {i+1}: {nonzero} 个变点被激活, δ 范围=[{deltas.min():.3f}, {deltas.max():.3f}]")
```

可能的输出：

```
路径 1: 3 个变点被激活, δ 范围=[-0.042, +0.087]
路径 2: 1 个变点被激活, δ 范围=[-0.015, +0.056]
路径 3: 4 个变点被激活, δ 范围=[-0.091, +0.034]
路径 4: 2 个变点被激活, δ 范围=[-0.023, +0.061]
路径 5: 3 个变点被激活, δ 范围=[-0.055, +0.072]
```

**每条的激活数量、位置、幅度都不同**——因为这些 δ 是随机抽的。这就是"1000 条不同路径"的来源。

---

#### 从 1000 条路径到预测区间

```
生成 1000 条趋势路径后:

第 30 天:
  路径1: trend=103    路径2: trend=108    ...
  路径3: trend=95     路径4: trend=112
  ...
  路径1000: trend=101

  排序 → 第 100 个 (10%分位) = yhat_lower[30]
          第 900 个 (90%分位) = yhat_upper[30]  (如果 interval_width=0.80)

对每一天重复 → 得到完整的 yhat_lower 和 yhat_upper 序列
```

**一句话总结**：趋势模拟 = 把历史变点的"随机性程度"（τ）外推到未来，每次外推时随机抽一组不同的 δ，1000 次抽样产生的1000 条不同趋势路径，它们的分散程度就代表了趋势的不确定性。

---

#### 为什么 τ 越大，区间越宽

现在你应该能直观理解这一点了：

```python
# τ = 0.01（紧先验）
deltas = np.random.laplace(0, 0.01, 25)
# → 大部分 δ ≈ 0 → 趋势各路径几乎一样 → 区间窄

# τ = 0.5（松先验）
deltas = np.random.laplace(0, 0.5, 25)
# → δ 可能抽到 ±1.0 甚至更大 → 趋势各路径差异巨大 → 区间宽
```

这正是 `changepoint_prior_scale` 的双重身份：**它在训练时控制趋势灵活性（第 3 章），在预测时控制不确定性宽度（本章）**——同一个 τ，两个作用。

```python
import numpy as np
import matplotlib.pyplot as plt

# 可视化：不同 uncertainty_samples 的效果
model = Prophet(uncertainty_samples=100)  # 先用 100 看个大概
model.fit(df)
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# yhat_lower 和 yhat_upper 就是模拟的分位数
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(forecast['ds'], forecast['yhat'], 'b-', label='预测')
ax.fill_between(forecast['ds'],
                forecast['yhat_lower'],
                forecast['yhat_upper'],
                alpha=0.2, label='80% 预测区间')
ax.legend()
```

### 7.2.3 `uncertainty_samples` 的选择

| 值 | 用途 | 代价 |
|-----|------|------|
| `0` | 关闭不确定性（只要点预测） | 无法得到 `yhat_lower` / `yhat_upper` |
| `100` | 快速探索、原型验证 | 区间边缘可能不够平滑 |
| `1000` (默认) | 日常使用 | 平衡点 |
| `2000+` | 需要非常平滑的区间、正式报告 | 计算时间增加 |

```python
# 对比不同采样数的区间平滑度
for samples in [10, 100, 1000]:
    m = Prophet(uncertainty_samples=samples)
    m.fit(df)
    fc = m.predict(m.make_future_dataframe(30))
    # 10 条路径 → 区间锯齿状
    # 1000 条路径 → 区间平滑
```

---

## 7.3 MCMC：获得完整的不确定性

### 7.3.1 背景：MCMC 是什么

在深入 Prophet 的 MCMC 用法之前，先理解这个技术本身——否则你只是在调 `mcmc_samples` 参数，不知道底下发生了什么。

---

**问题：我们想要什么？**

贝叶斯推断的目标是得到**后验分布** $P(\theta \mid \text{data})$——在看到数据之后，参数 $\theta$ 长什么样。

用第 3 章的先验/后验框架重述：

$$P(\theta \mid \text{data}) \propto \underbrace{P(\text{data} \mid \theta)}_{\text{似然}} \times \underbrace{P(\theta)}_{\text{先验}}$$

问题来了：对于 Prophet 这种有几十个参数的模型，后验分布是一个**几十维空间中的复杂曲面**。你没法像画一张二维图那样把它"画出来"，更没法直接从中抽样。

---

**直觉：一个在迷雾中登山的人**

想象你被空投到一片完全陌生的山区，浓雾弥漫，你看不到整片地形。你的任务是**搞清楚这片山区哪里高哪里低**——即后验分布的形状。

你能做的只有一件事：**迈出一步，感受脚下是上坡还是下坡**。

```
MCMC 的核心直觉：

     ╱╲
    ╱  ╲      ← 后验分布（真值，但你看不到全貌）
   ╱    ╲
  ╱      ╲
╱╱        ╲╲
──────────────

你是一只在迷雾中行走的蚂蚁：
  ● → ● → ● → ●
      ↑"这里更高，可能接近山顶了"
            ↓"在下坡，继续走"
                  ● → ● → ...
```

这就是 **M**arkov **C**hain **M**onte **C**arlo 的三个词逐一拆解：

---

#### Monte Carlo（蒙特卡洛）

用随机采样来近似一个难以直接计算的量。

最经典的例子——估算 π：

```
在一个 2×2 的方形内随机撒点，落在内切圆里的比例 × 4 ≈ π

        ┌──────────┐
        │  ●  ●●● ●│
        │ ● ● ● ●  │    撒 10000 个随机点
        │●●   ● ●● │    圆内点数 / 总点数 × 4 ≈ 3.141...
        │ ● ●● ● ● │
        │  ● ●  ●  │
        └──────────┘
```

对于 Prophet 来说："量" = 后验分布 $P(\theta \mid \text{data})$。"采样" = 生成很多组参数 $\theta_1, \theta_2, ..., \theta_N$，用这 N 组参数的分布来近似真实后验。

---

#### Markov Chain（马尔可夫链）

但你怎么保证撒的"随机点"真的集中在后验概率高的地方，而不是在低概率区域浪费？你不能直接在后验分布上撒点——因为你根本不知道它长什么样。

马尔可夫链解决了这个问题：**每一步只依赖于上一步**。

```
第 1 步：随机选一个起点 θ₁
第 2 步：从 θ₁ 出发，随机试探一个附近的 θ_proposal
         ├── 如果 θ_proposal 更"好"（后验概率更高）→ 接受，θ₂ = θ_proposal
         └── 如果 θ_proposal 更"差" → 以一定概率接受（概率 = 好/差 之比）
第 3 步：从 θ₂ 出发，重复...
...
```

关键设计：**"以一定概率接受更差的点"**——这让链偶尔能爬出局部最优，探索更广阔的区域。

```
后验概率
  │        ╱╲
  │       ╱  ╲         ← 真正的山顶
  │   ╱╲╱    ╲
  │  ╱  ╲     ╲
  │ ╱    ╲──── ╲___   ← 局部小山坡
  └───────────────────
        ↑
     不从最高点开始    → 链随机游走，逐步"发现"最高区域
```

经过足够多的步数（称为 burn-in 期），链会收敛——它花在每个区域的时间，与该区域的后验概率成正比。高概率的地方访问得多，低概率的地方访问得少。

---

#### 串起来：MCMC = 用一条随机游走的链来近似复杂的后验分布

```
MCMC 的工作流：

    1. 随机初始化 θ
           │
    2. 循环 N 次：
       ├── 提议一个新的 θ*（基于当前 θ 做微小扰动）
       ├── 计算接受概率 α = min(1, P(θ*|data) / P(θ|data))
       ├── 以概率 α 接受 θ*，否则保留 θ
       └── 记录当前的 θ
           │
    3. 扔掉前 B 个样本（burn-in 期，链还没收敛）
           │
    4. 剩下的 N-B 个样本 ≈ 来自真实后验的样本
```

用得到的样本，你可以回答：
- "参数最可能的值是多少？" → 样本的中位数
- "参数的不确定性有多大？" → 样本的 2.5% ~ 97.5% 分位数
- "两个参数有相关性吗？" → 画样本的散点图

---

#### 为什么 Prophet 默认不用 MCMC？

| | MAP | MCMC |
|------|-----|------|
| **原理** | 找一个点：后验的最高峰 | 描一条路：在后验曲面上游走 |
| **速度** | 快（优化一个目标函数） | 慢（需要几千步游走才收敛） |
| **输出** | 一组"最优"参数 | 参数的后验分布（完整的不确定性） |
| **适用** | 数据多、噪声小的日常场景 | 数据少、噪声大、需要精确概率的场景 |

Prophet 的设计哲学是"让预测民主化"——默认用 MAP，保证速度和易用性。MCMC 留给需要高精度不确定性量化的专业场景。

---

### 7.3.2 MAP 的局限

默认的 MAP 模式有一个重要限制：**它只对趋势做不确定性模拟，季节性和噪声参数被"钉"在了 MAP 估计值上。**

这意味着默认的预测区间**可能偏窄**——它忽略了"季节性估计可能不准"这一事实。

### 7.3.3 MCMC 在 Prophet 中做了什么

开启 MCMC 后，Prophet 不再只找"最可能"的参数，而是用上面描述的马尔可夫链在参数空间中游走，**从后验分布中采样**：

```
MAP:   找一组最佳参数 θ̂ → 只模拟趋势不确定性
MCMC:  从 P(θ | data) 中采样 1000 组参数 → 每组参数都模拟趋势
       → 完整的不确定性：参数不确定 + 趋势不确定 + 观测噪声
```

```python
# 开启 MCMC（需要安装 pystan 或用 cmdstanpy 后端）
model = Prophet(
    mcmc_samples=300,        # MCMC 采样数（默认 0 = 关闭）
    uncertainty_samples=500, # 每组参数做多少次趋势模拟
)
```

### 7.3.3 何时需要 MCMC

| 场景 | 建议 | 理由 |
|------|------|------|
| 数据量充足（> 500 点）、模式稳定 | MAP 够用 | 参数估计已经比较准，额外不确定小 |
| 数据量较少、噪声大 | MCMC | 参数估计本身就不确定，MAP 区间偏窄 |
| 需要完整的概率分布 | MCMC | 安全库存计算、风险量化等场景 |
| 快速迭代、原型验证 | MAP | MCMC 慢 10-50 倍 |

### 7.3.4 MCMC 的代价

```python
# MCMC 的速度差异
import time

# MAP — 秒级
t0 = time.time()
model_map = Prophet()
model_map.fit(df)
print(f"MAP: {time.time() - t0:.1f}s")

# MCMC — 分钟级（取决于数据量和采样数）
t0 = time.time()
model_mcmc = Prophet(mcmc_samples=300)
model_mcmc.fit(df)
print(f"MCMC: {time.time() - t0:.1f}s")
```

> ⚠️ 如果你在 Windows 上使用 `pystan` 遇到问题，Prophet 现在推荐 `cmdstanpy` 作为 Stan 后端，安装：`uv pip install cmdstanpy`，然后用 `Prophet(stan_backend='CMDSTANPY')`。

---

## 7.4 预测区间的正确解读

### 7.4.1 区间不是"置信区间"，是"预测区间"

| 名称 | 含义 | Prophet 给的是这个吗？ |
|------|------|----------------------|
| **置信区间 (Confidence Interval)** | "均值"的不确定性——如果重做 100 次实验，均值在哪里 | ❌ |
| **预测区间 (Prediction Interval)** | "单个未来观测值"的不确定性——包含了数据本身的随机波动 | ✅ |

Prophet 的 `yhat_lower` / `yhat_upper` 是**预测区间**——它考虑了趋势不确定性和观测噪声。这意味着**80% 区间的含义是：未来某一个具体日期的实际值，有 80% 的概率落在这个区间内。**

### 7.4.2 区间的典型形态

```
预测区间的"喇叭口"形状:

 历史           预测
  │    ╱╲        │       ╱╲
  │   ╱  ╲       │      ╱  ╲
  │  ╱    ╲      │     ╱    ╲
  │╱╱      ╲╱    │   ╱╱      ╲╱
  └────────────  └─╱───────────╲──
                   ╱  区间越来越宽 ╲
                  由于趋势不确定性累积
```

**越远的预测越不确定**——这是时间序列预测的通用规律。区间宽度大致随预测 horizon 的平方根增长（趋势不确定累积），但如果 `changepoint_prior_scale` 很大，宽度增长更快。

### 7.4.3 区间宽度由什么决定

| 因素 | 对区间宽度的影响 |
|------|-----------------|
| 历史数据的噪声水平（σ） | 噪声越大，区间越宽 |
| 历史趋势变化幅度 | 历史转折越多，未来区间越宽 |
| `changepoint_prior_scale` | 越大 → 允许更多未来变化 → 区间越宽 |
| `interval_width` | 越宽 → 区间越宽（80% → 95% 宽度增加约 30%） |
| `uncertainty_samples` | 不影响宽度，只影响平滑度 |
| 预测 horizon | 越远越宽 |

```python
# 对比不同 interval_width
model = Prophet(interval_width=0.95)  # 95% 区间 → 更宽，更"保守"
```



---

## 7.5 实操：不确定性驱动的决策

### 7.5.1 安全库存计算

```python
# 用预测区间下界计算安全库存
# 90% 的服务水平：保证 90% 情况下不缺货
forecast['safety_stock'] = forecast['yhat'] - forecast['yhat_lower']
# 如果 yhat_lower 用的是 80% 区间，考虑用更宽的区间
```

### 7.5.2 异常检测

```python
# 实际值超出预测区间 → 可能是异常
forecast['is_anomaly'] = (df['y'] > forecast['yhat_upper']) | \
                          (df['y'] < forecast['yhat_lower'])
print(f"异常天数: {forecast['is_anomaly'].sum()} / {len(forecast)}")
```

### 7.5.3 区间覆盖率验证

```python
# 检查历史数据的区间覆盖率
# 对于 80% 区间，理想覆盖率应接近 80%
def check_coverage(y_true, y_lower, y_upper):
    in_interval = (y_true >= y_lower) & (y_true <= y_upper)
    return in_interval.mean()

coverage = check_coverage(df['y'], forecast['yhat_lower'], forecast['yhat_upper'])
print(f"历史区间覆盖率: {coverage:.1%}")
# 如果 coverage 远低于 interval_width → 模型过于自信，区间太窄
# 如果 coverage 远高于 interval_width → 模型过于保守，区间太宽
```

---

## 7.6 常见问题速查

| 问题 | 原因 | 解决 |
|------|------|------|
| 预测区间几乎贴着 yhat | `uncertainty_samples=0` 或数据极平稳 | 检查 `uncertainty_samples` ≥ 100 |
| 预测区间极宽，毫无参考价值 | 历史噪声大 + 预测 horizon 太长 | 缩短预测期，或调小 `changepoint_prior_scale` |
| MCMC 报错或极慢 | pystan 兼容问题或采样数太多 | 换 `cmdstanpy` 后端，减少 `mcmc_samples` |
| `yhat_lower` 出现负值（销量预测） | 正态假设允许负值 | 对 y 做 log 变换，或改用逻辑增长 |
| 历史数据覆盖率远低于 `interval_width` | 模型过于自信（区间偏窄） | 开启 MCMC，或调大 `changepoint_prior_scale` |

---

## 7.7 核心概念清单

| 概念 | 一句话理解 |
|------|-----------|
| **MAP** | 找"最可能"的那组参数——快，但忽略参数不确定性 |
| **MCMC** | 从参数的后验分布中采样很多组参数——慢，但给出完整不确定性 |
| **预测区间** | 未来"单个观测值"的合理范围——比置信区间更宽 |
| **uncertainty_samples** | 趋势模拟的路径数——影响区间平滑度，不影响宽度 |
| **interval_width** | 区间覆盖概率——影响宽度 |
| **喇叭口效应** | 区间随预测 horizon 扩散——时间序列表征 |

### 参数速查

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `interval_width` | `0.80` | 预测区间宽度（80% = 默认，95% = 更保守） |
| `uncertainty_samples` | `1000` | 趋势模拟路径数（0 = 关闭不确定性） |
| `mcmc_samples` | `0` | MCMC 采样数（0 = 只用 MAP） |

---

下一章进入模型诊断与交叉验证——如何科学地判断你的模型做得好不好。
