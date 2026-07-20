# DeepAR: Probabilistic Forecasting with Autoregressive RNNs

> **论文**: [DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks](https://arxiv.org/abs/1704.04110)
> **机构**: Amazon Research (2017, 2019 正式发表在 *International Journal of Forecasting*)
> **作者**: David Salinas, Valentin Flunkert, Jan Gasthaus, Tim Januschowski

---

## 1. 核心思想

DeepAR 解决的核心问题：**给定多条时间序列的历史值，预测每条序列未来值的概率分布**。

与传统的点预测 (point forecast) 不同，DeepAR 输出的是**预测分布**（如高斯分布的 μ 和 σ），天然支持不确定性量化。

一句话概括：**用一个全局的 LSTM 网络对所有时间序列联合建模，每条序列的预测建立在其它序列学习到的模式上（通过 cross-learning），输出的是分布参数而非单一数值。**

---

## 2. 时间序列预测方法的演进路线

```
传统统计方法 (ARIMA, ETS, Theta)
        ↓
机器学习方法 (GBDT with feature engineering, XGBoost/LightGBM)
        ↓
全局深度学习方法 (DeepAR, N-BEATS, Informer, PatchTST, TimesFM...)
```

| | 传统统计 | GBDT 机器学习 | DeepAR (深度全局模型) |
|---|---|---|---|
| **学习方式** | 每条序列独立拟合 | 手工特征 + 回归 | 全局联合学习 |
| **冷启动** | 需要历史数据 | 需要特征工程 | 天然支持 (item embedding) |
| **不确定性** | 理论置信区间 | 分位数回归/残差 | 直接输出分布参数 |
| **多序列规模** | O(N×M) 个模型 | O(M) 个模型 | O(1) 个模型 |
| **非线性模式** | 弱 | 中等 (树模型局限) | 强 (LSTM 记忆) |

---

## 3. 模型架构

### 3.1 整体结构

```
输入: z_{t-1} (上一步观测值) ──┐
                                ├──→ LSTM Cell ──→ h_t ──→ 线性层 ──→ θ_t (分布参数)
输入: x_t (协变量/时间特征) ────┘
```

DeepAR 本质是一个**条件概率模型**：$$P(\mathbf{z}_{i, t_0:T} \mid \mathbf{z}_{i, 1:t_0-1}, \mathbf{x}_{i, 1:T})$$

其中 $z_{i,t}$ 表示序列 $i$ 在时间 $t$ 的值，$x_{i,t}$ 是已知的协变量（如星期几、节假日、序列 ID 等）。

### 3.2 关键组件

| 组件 | 作用 | 细节 |
|---|---|---|
| **LSTM 编码器** | 将历史序列编码为隐状态 | 多层 LSTM，多层之间的 dropout |
| **分布头 (Distribution Head)** | 从隐状态 $h_t$ 映射到分布参数 | 高斯分布 → μ, σ；负二项分布 → μ, α |
| **Item Embedding** | 学习每条序列的"身份向量" | 类似用户 embedding，解决冷启动 |

### 3.3 前向计算流程

```
for t in range(history_len):
    z = scale(z_{t-1})              # 可选: 输入缩放
    x = [covariates_t, item_emb]    # 拼接协变量和 embedding
    h_t = LSTM([z, x], h_{t-1})     # LSTM 前向
    θ_t = FC(h_t)                   # 从隐状态计算分布参数
    loss += -log P(z_t | θ_t)       # 负对数似然

# 这里的 θ_t = (μ_t, σ_t) 用于高斯分布
# 训练时每一步的 z_{t-1} 使用真实值 (teacher forcing)
```

### 3.4 数据流概览

以电商 1000 个 SKU 日销量预测为例（$C=56$, $H=7$, $d_h=64$, $d_e=16$, $d_c=3$, $B=64$，负二项分布）：

**训练**（Teacher Forcing）：
1. Scale Normalization → 拼接 `[z_{t-1}, cov_t, emb_i]` 得到 `(64, 56, 20)` 维输入
2. LSTM → `lstm_out: (64, 56, 64)` → FC → `mu, alpha: 各 (64, 56)` → NLL loss
3. Teacher Forcing 偏移：$t$ 时刻输入 $z_{t-1}^{real}$（真实值），输出分布参数预测 $z_t$

**推理**（祖先采样）：
1. 历史序列编码为最终隐状态 $(h_n, c_n)$
2. 逐步采样 $K=200$ 条路径：$z_t \sim P(z | \mu_t, \alpha_t)$，$z_t$ 回送下一步
3. `samples: (200, 7)` → 取 P10/P50/P90 → 乘以 `scale` 反归一化 → 业务决策

> 完整的数据流转和每一步 shape 变化见 `code/deepar_demo.py` Part B 的注释。

---

## 4. 似然函数 (Likelihood Function)

DeepAR 的核心"损失函数"就是**负对数似然 (NLL, Negative Log-Likelihood)**。

### 4.1 高斯似然 (适用于连续值)

$$P(z \mid \mu, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(z - \mu)^2}{2\sigma^2}\right)$$

$$\text{NLL} = \frac{1}{2}\log(2\pi) + \log(\sigma) + \frac{(z - \mu)^2}{2\sigma^2}$$

- **μ 预测**: 线性层直接输出
- **σ 预测**: 通过 `softplus` 保证正值: `σ = softplus(raw_σ)`

### 4.2 负二项似然 (适用于计数/离散值，如销量)

$$P(z \mid \mu, \alpha) = \frac{\Gamma(z + 1/\alpha)}{\Gamma(z + 1)\Gamma(1/\alpha)} \left(\frac{1}{1 + \alpha\mu}\right)^{1/\alpha} \left(\frac{\alpha\mu}{1 + \alpha\mu}\right)^z$$

- **μ 预测**: `softplus(raw_μ)` 保证正值
- **α 预测**: `softplus(raw_α)` — 这是 dispersion parameter

### 4.3 其它可选分布

| 分布类型 | 适用场景 | 参数 |
|---|---|---|
| Gaussian | 连续值，有正有负 | μ, σ |
| Student-t | 连续值，厚尾 | μ, σ, ν |
| Negative Binomial | 非负计数 | μ, α |
| Beta | (0,1) 区间值 | α, β |
| Categorical | 有限离散值 | p₁, p₂, ..., pₖ |

---

## 5. 训练过程

### 5.1 数据预处理

```python
# 1. 构建训练样本: 从每条序列中随机采样窗口
for each series i:
    sample a random window t_start...t_start+context_len+pred_len
    past = series[t_start : t_start+context_len]   # 历史
    future = series[t_start+context_len : ...]       # 待预测

# 2. 可选: 对每条序列做 scale 归一化
scale_i = mean(abs(past))  # 或 median
z_scaled = z / scale_i
```

### 5.2 Teacher Forcing

训练时，LSTM 每步输入的是**上一步的真实观测值**（而非上一步自己的预测），这称为 Teacher Forcing：

```
t=1: LSTM(z_0_real, x_1) → θ_1, 计算 loss(z_1_real | θ_1)
t=2: LSTM(z_1_real, x_2) → θ_2, 计算 loss(z_2_real | θ_2)  ← 关键: 用真实值，不是采样值
t=3: LSTM(z_2_real, x_3) → θ_3, 计算 loss(z_3_real | θ_3)
...
```

### 5.3 训练技巧

| 技巧 | 说明 |
|---|---|
| **Scale normalization** | 每条序列除以其历史均值的绝对值，稳定梯度 |
| **随机窗口采样** | 每个 epoch 从每条序列中随机取不同窗口，增强泛化 |
| **Gradient clipping** | LSTM 训练常用，防止梯度爆炸 |
| **Early stopping + LR scheduler** | 标准做法 |
| **辍学 (Dropout)** | LSTM 层间 + 输入层 dropout |

---

## 6. 预测/推理

训练完成后，DeepAR 通过**祖先采样 (Ancestral Sampling)**生成多条预测路径：

```
for each sample path k in 1..K:
    h = h_last  (训练好的最后一个隐状态)
    z = z_last  (最后一个已知观测值)

    for t in 1..prediction_length:
        θ_t = FC(LSTM([z, x_t], h))     # 前向计算分布参数
        z ~ P(z | θ_t)                  # 从分布中采样
        h = LSTM 隐状态更新
        predictions[k][t] = z

# 最终得到 K 条采样路径，可以计算:
# - 中位数 (P50) 作为点预测
# - 分位数 (P10, P90) 构建预测区间
```

**与训练的关键区别**：推理时 LSTM 每步的输入来自上一步的**采样值**（而非真实值），这引入了长期预测的不确定性积累。

---

## 7. 关键创新点

### 7.1 冷启动 (Cold Start) — Item Embedding

对于**没有历史数据的新产品**，DeepAR 可以通过 item embedding 学到"同类产品"的模式：

```python
# 训练时
item_emb = nn.Embedding(num_items, emb_dim)  # 每个 item 学一个向量
h_0 = init_state_from_embedding(item_emb)    # 用 embedding 初始化 LSTM 起始状态

# 冷启动预测
new_item_emb = avg(同类历史产品的 embedding)  # 或从特征映射
prediction = model.predict(new_item_emb)
```

### 7.2 概率输出

不是预测一个值，而是预测一个分布。这天然支持：
- **分位数预测**: P10, P50, P90
- **概率决策**: "库存不足的概率" → 决策进货量
- **多场景分析**: K 条采样路径模拟 K 种可能的未来

### 7.3 跨序列学习 (Cross-learning)

传统 ARIMA 对每条序列单独建模，DeepAR 一个模型同时学所有序列，将高方差序列的信息"借"给低方差序列。

### 7.4 自动处理多频率

由于用的是 LSTM 而非频率相关的分解（如 STL），同一模型可以处理不同采样频率的数据（只需协变量标明时间特征）。

---

## 8. 优缺点与适用场景

### 优点
- ✅ 概率输出，天然不确定性量化
- ✅ 冷启动能力强（item embedding）
- ✅ 一个模型处理成千上万条序列
- ✅ 可叠加多种协变量（时间、类别、外部特征）
- ✅ 效果在 M4/M5 比赛中得到验证

### 缺点
- ❌ 长序列计算慢（LSTM 是顺序的）
- ❌ 训练时间较长（需要很多序列样本）
- ❌ 对超参数敏感（隐层大小、学习率等）
- ❌ 极长预测窗口 (>100) 可能样本退化

### 适用场景
- 🟢 电商：万级 SKU 需求预测
- 🟢 能源：千家用户的用电预测
- 🟢 金融：多资产波动率预测
- 🟢 供应链：多仓库库存预测

---

## 9. 2026 年回望：DeepAR 在深度学习时序生态中的定位

> 2026 年，距离 DeepAR 诞生已近十年。时序预测领域经历了 Transformer 浪潮、线性模型复兴、大模型涌现、Mamba/SSM 入侵，格局已翻天覆地。本节站在 2026 年回望，给出一个不回避优点也不掩饰局限的诚实定位。

---

### 9.1 快照：2026 年深度学习时序预测的四条技术路线

```
                        时序深度学习模型版图 (2026)
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │  RNN 自回归家族    │  │ Transformer 家族  │  │  线性/MLP 复兴    │  │
│  │                  │  │                  │  │                  │  │
│  │ DeepAR (2017)    │  │ Informer (2021)  │  │ DLinear (2023)   │  │
│  │ DeepState (2018) │  │ Autoformer(2021) │  │ NLinear (2023)   │  │
│  │ MQ-RNN (2017)    │  │ PatchTST (2023)  │  │ TiDE (2023)      │  │
│  │                  │  │ iTransformer(24) │  │ N-BEATS (2020)   │  │
│  │                  │  │ Crossformer(23)  │  │                  │  │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘  │
│           │                     │                     │             │
│  ┌────────┴─────────────────────┴─────────────────────┴─────────┐  │
│  │                    第四波：预训练大模型                         │  │
│  │                                                                │  │
│  │  TimesFM  (Google, 2024)    — Decoder-only, 零样本预测         │  │
│  │  MOIRAI   (Salesforce, 2024)— 多领域统一模型，千万参数         │  │
│  │  Chronos  (Amazon, 2024)    — 语言模型思路 tokenize 时序       │  │
│  │  时序 GPT (各厂, 2025-)     — 微调即用，省去从头训练           │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │            第五波（2025-2026 新兴）：Mamba/SSM                 │  │
│  │  S-Mamba, TimeMachine, MambaTS  — 线性复杂度 + Transformer   │  │
│  │  级建模能力；在长序列任务上对 Transformer 有竞争力            │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 9.2 四条路线的本质区别和代表模型对比

| 路线 | 核心思想 | 代表模型 | 概率输出 | 长序列 | 冷启动 | 2026 地位 |
|------|---------|---------|---------|--------|--------|----------|
| **RNN 自回归** | 逐步条件概率 $P(z_t \mid z_{<t})$ | DeepAR | ✅ 原生 | ❌ | ✅ | 稳定、务实 |
| **Transformer** | Patch 化 + 全局自注意力 | PatchTST, iTransformer | ⚠️ 需加头 | ✅ | ⚠️ | 学术主流 |
| **线性/MLP 复兴** | "简单的就是最好的" | DLinear, N-BEATS | ❌ | ✅ | ❌ | 强 baseline |
| **预训练大模型** | 海量数据预训练 + zero/few-shot | TimesFM, MOIRAI | ⚠️ 仅分位数 | ✅ | ✅ | 最热方向 |
| **Mamba/SSM** | 状态空间模型，线性复杂度 | S-Mamba | ⚠️ 探索中 | ✅ | ⚠️ | 有潜力 |

### 9.3 逐模型详细对比（更新至 2026）

| 模型 | 架构 | 输出 | 核心创新 | 概率？ | 冷启动？ | 长序列效率 | 年份 |
|------|------|------|---------|--------|---------|-----------|------|
| **DeepAR** | LSTM 自回归 | μ,σ (分布) | 全局+概率+embedding | ✅ | ✅ | 低 ($O(C)$) | 2017 |
| **DeepState** | LSTM+线性SSM | 分布 | 结构化趋势/季节 | ✅ | ❌ | 低 | 2018 |
| **N-BEATS** | 全连接堆叠 | 点预测 | 基扩展+双重残差 | ❌ | ❌ | 中 | 2020 |
| **Informer** | Transformer | 点预测 | ProbSparse 注意力 | ❌ | ❌ | 高 ($O(T\log T)$) | 2021 |
| **Autoformer** | Transformer | 点预测 | 自相关替代自注意力 | ❌ | ❌ | 高 | 2021 |
| **DLinear** | 线性层 | 点预测 | 极简：就是一层 Linear | ❌ | ❌ | 极高 ($O(T)$) | 2023 |
| **PatchTST** | Transformer | 点预测 | 将序列切成 patch 再 attention | ❌ | ❌ | 高 | 2023 |
| **iTransformer** | Transformer | 点预测 | 倒置：变量维度做 attention | ❌ | ❌ | 高 | 2024 |
| **TimesFM** | Decoder-only | 点+分位数 | 预训练 100B 时间点 | ⚠️ | ✅ (预训练) | 高 | 2024 |
| **MOIRAI** | Masked Encoder | 分布 | 多领域统一预训练 | ⚠️ | ✅ (预训练) | 高 | 2024 |
| **Chronos** | T5-based | 分布 | tokenize 数值为离散词 | ⚠️ | ✅ (预训练) | 中 | 2024 |
| **S-Mamba** | Mamba SSM | 点预测 | 状态空间替代注意力 | ❌ | ❌ | 极高 ($O(T)$) | 2025 |

### 9.4 DeepAR 在 2026 年的「不可替代」和「已被超越」

**🔴 已被超越的部分：**

| 维度 | DeepAR 的情况 | 谁做得更好 |
|------|-------------|-----------|
| **纯点预测精度** | 在大多数 benchmark 上已被超越 | DLinear, PatchTST, iTransformer 通常在 MSE/MAE 上高出 5-20% |
| **长序列效率** | LSTM 串行，$C>200$ 训练很慢 | Transformer（并行）、Mamba（线性）、DLinear（$O(T)$）都更快 |
| **序列长度建模** | LSTM 的梯度在 100+ 步后衰减明显 | PatchTST 通过 patch 化有效捕捉长程依赖 |
| **学术热度** | 2017 年的工作，论文已不是引用热点 | iTransformer、MOIRAI、Chronos 是当前热点 |
| **零样本能力** | 无—每个场景需从头训练 | TimesFM、Chronos 等预训练模型可 zero-shot 预测 |

**🟢 至今仍不可替代的部分：**

| 维度 | DeepAR 的独特价值 | 为什么仍然重要 |
|------|-----------------|---------------|
| **原生概率输出** | 从架构设计之初就是分布预测，不是后装的 | 大多数 Transformer 模型（PatchTST, iTransformer, DLinear）以点预测为默认，概率预测是"后加"的。DeepAR 的 NLL 训练 + 祖先采样让概率预测是**一等公民** |
| **冷启动** | Item embedding 为冷启动而生 | 预训练大模型虽然也支持零样本，但在领域内（如同平台内新 SKU）DeepAR 的 embedding 学得更精细 |
| **数据效率** | 不需要海量预训练数据，几百条序列即可 | 大模型需要 100B+ 时间点预训练；DeepAR 可以仅用你自己的数据训练 |
| **工程成熟度** | 在 Amazon 生产系统跑了近 10 年 | GluonTS 极其成熟，有完整的评估/回测/部署工具链；大模型还在快速迭代中 |
| **简单可控** | 架构简单，调参经验丰富，问题好排查 | 大模型是黑盒，出了问题很难定位 |
| **GluonTS 生态** | 一整套从数据加载到部署的工程方案 | 大多数新模型只有论文代码，和 DeepAR 共享 GluonTS 生态 |

### 9.5 关键洞察：LSTM 不是 DeepAR 的包袱

一个常见的误解是"LSTM 过时了，所以 DeepAR 也过时了"。但仔细想：

- **架构不是瓶颈**：对绝大多数业务场景（序列 50~300 步，几千条序列），LSTM 的计算效率完全够用，且 GPU 利用率高
- **概率自回归的范式才是核心**：用 Transformer 换掉 LSTM 保持同样的概率自回归框架，就是后来很多工作的做法。但范式本身是 DeepAR 定义的
- **DLinear 的冲击被夸大了**：DLinear 在特定 benchmark（ETT, Weather）上表现好，但在**高噪声、多序列异质性**的场景（如零售万级 SKU）中，简单的线性模型无法捕捉序列间的共享模式。DeepAR 的 cross-learning 在这些场景仍显著优于线性 baseline

> **一句话定位**：DeepAR 在 2026 年不是 SOTA（State-of-the-Art），但它是 **SOP（State-of-Practice）**——工程上最可靠、最成熟、最被验证的概率预测方案。

### 9.6 实际选型决策树（2026 版）

```
你的场景是什么？
│
├── 只有 1 条序列，没有其他相关序列
│   └── → 别用 DeepAR。试试 N-BEATS / DLinear / 传统统计方法
│
├── 有 100+ 条相关序列，需要概率输出（分位数/预测区间）
│   └── → DeepAR 是首选，尤其当：
│           - 序列间量级差异大 → item embedding 天然处理
│           - 有新序列需要冷启动 → embedding 天然支持
│           - 需要生产部署 → GluonTS 最成熟
│
├── 有 100+ 条相关序列，只需要点预测，追求极致精度
│   └── → PatchTST / iTransformer 作为主模型
│           DeepAR 作为概率 baseline
│
├── 需要预测极长序列（>500 步）
│   └── → DeepAR 不合适。换 PatchTST / Mamba 类 / 大模型
│
├── 没有训练数据 / 需要快速出结果
│   └── → TimesFM / Chronos zero-shot（但注意领域适配风险）
│           DeepAR 需要训练，不适合此场景
│
├── 需要概率输出 + 长序列
│   └── → 先用 DeepAR 做 baseline（数据效率最高）
│           再试预训练大模型的概率变体（如 MOIRAI）
│
└── 研究/发论文
    └── → DeepAR 已不是好的 baseline。用 DLinear/PatchTST/iTransformer 做 baseline
```

### 9.7 未来展望：DeepAR 的思想会以什么形式延续？

1. **预训练大模型的概率化**：Chronos、MOIRAI 已经在朝概率输出方向走，但它们本质上是在学习 DeepAR 开创的"全局概率自回归"范式的预训练版本
2. **GluonTS 生态吸收新架构**：GluonTS 已经支持了 Transformer-based 模型和 Lag-Llama 等，DeepAR 作为生态中的一个模型长期共存
3. **混合方案**：在工业界的趋势是用 DeepAR 提供稳定基线 + 大模型做 scenario planning / what-if 分析，各有分工而非替代
4. **Mamba/SSM + 概率头**：将 DeepAR 的概率头嫁接到 Mamba 骨干上，可能同时获得线性复杂度和原生概率输出——这是个值得关注的潜在方向

> **最后总结**：DeepAR 在 2026 年不酷，但很稳。它不是你在论文里引用的对象，但它是让你老板睡得好觉的模型。学术界关心是不是 SOTA，工业界关心是不是 SOP——DeepAR 是后者。

---

## 10. 评价指标

论文推荐使用 **ρ-risk (quantile loss)** 来评估概率预测：

$$\text{QL}_{\rho}(y, \hat{y}_\rho) = \begin{cases} \rho \cdot (y - \hat{y}_\rho), & y \geq \hat{y}_\rho \\ (1 - \rho) \cdot (\hat{y}_\rho - y), & y < \hat{y}_\rho \end{cases}$$

其中 $\hat{y}_\rho$ 是预测的第 $\rho$ 分位数。

常用的具体指标：
- **ND (Normalized Deviation)**: 点预测误差 / 基准误差
- **RMSE / MAE**: 点预测标准指标
- **CRPS (Continuous Ranked Probability Score)**: 概率预测整体校准度
- **P50 loss / P90 loss**: 分位数损失

---

## 11. 实践建议

1. **数据量要求**: 至少几百条序列，每条序列几十到几百个时间步
2. **预处理**: 按序列做 scale normalization，训练前去除异常值
3. **超参数**: LSTM 2-3 层，隐藏维度 40-200，embedding 维度 10-50
4. **采样数**: 推理时 K=100~1000 条采样路径可以得到稳定分位数
5. **协变量**: 至少加入一天中的小时/一周中的天/月份等时间特征
6. **批量大小**: 序列条数 × 窗口数，一般 64-256 较为合适
7. **GPU**: 几百条序列用 CPU 也行；上万条序列建议 GPU

---

## 12. 生产级工具链

### 12.1 GluonTS（Amazon 官方，首选）

```bash
pip install gluonts
```

GluonTS 是 Amazon 官方的时间序列预测库，DeepAR 就诞生于此。现已支持 PyTorch 后端。

```python
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.distributions import StudentTOutput, NegativeBinomialOutput

estimator = DeepAREstimator(
    freq="D",
    prediction_length=28,
    context_length=56,
    hidden_size=64,
    num_layers=2,
    distr_output=StudentTOutput(),  # 连续值推荐 StudentT
    # distr_output=NegativeBinomialOutput(),  # 计数数据用这个
)

predictor = estimator.train(train_dataset)
```

### 12.2 其他可选的 DeepAR 实现

| 库 | 后端 | 特点 |
|---|---|---|
| **GluonTS** | PyTorch / MXNet | Amazon 官方，最完整 |
| **pytorch-forecasting** | PyTorch Lightning | DeepAR 作为子模型，自带超参搜索 |
| **Darts** | PyTorch / sklearn | 统一 API，方便和其他模型对比 |

### 12.3 分布选择决策树

```
数据是连续值？
  ├── 可能有异常值 → StudentTOutput()
  ├── 无异常值     → GaussianOutput()
  └── (0,1) 区间  → BetaOutput()
数据是非负整数（如销量）？
  └── → NegativeBinomialOutput()
```

---

## 13. 进一步阅读

- **原论文**: [DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks](https://arxiv.org/abs/1704.04110) (2017)
- **Journal 版**: *International Journal of Forecasting*, 2020, 36(3): 1181-1191
- **GluonTS 官方文档**: https://ts.gluon.ai/ — API 文档 + 教程 Notebook
- **GluonTS GitHub**: https://github.com/awslabs/gluon-ts
- **本例代码**: `code/deepar_demo.py` — GluonTS 生产实现 + PyTorch 教学实现
- **M5 Competition**: DeepAR 在 M5 预测竞赛中获得优异成绩
- **pytorch-forecasting**: https://pytorch-forecasting.readthedocs.io/ — 另一套 PyTorch Lightning 封装

---

## 14. 面试高频问题与最优回答

> 以下回答按 **"一句话结论 → 展开解释 → 加分细节"** 三层结构组织，面试时根据时间灵活调整深度。

---

### Q1: DeepAR 和传统 ARIMA 的本质区别是什么？

**一句话**：ARIMA 为每条序列单独建模，DeepAR 用一个全局模型同时学所有序列。

**展开**：

- ARIMA 是**局部模型**：对每条时间序列分别拟合 $p, d, q$ 参数，序列之间没有信息共享。1000 条序列需要 1000 个模型。
- DeepAR 是**全局模型**：一个 LSTM 网络对所有序列联合训练，1000 条序列共享同一套参数，通过序列间的共同模式（季节性、趋势形态）互相"借力"。
- 结果：DeepAR 在处理大量相关序列（如万级 SKU）时远优于 ARIMA，尤其对**历史短的序列**（冷启动场景），因为模型可以从相似序列中迁移知识。

**加分**：可以提 "cross-learning" 概念——DeepAR 把每条序列视为高维分布的独立抽样，LSTM 学到的是这些抽样背后的共享生成过程。

---

### Q2: DeepAR 的损失函数是什么？为什么不用 MSE？

**一句话**：损失函数是**负对数似然 (NLL)**，不用 MSE 是因为 DeepAR 输出的是**分布**而非单点。

**展开**：
- MSE 隐含假设是高斯分布且方差为常数（$\mathcal{L}_{MSE} \propto -\log P(z|\mu, \sigma=const)$），这丢失了不确定性信息。
- NLL 允许模型同时学习均值和方差（或 dispersion），预测不仅告诉你"可能是多少"，还告诉你"有多大的不确定性"。
- 比如销量预测：卖 10 件 ± 2 件和卖 10 件 ± 20 件，MSE 给出相同的点预测损失，NLL 会惩罚"预测波动大但实际很稳定"的情况。

**加分**：可以补充 NLL 的数学形式 —$$\text{Gaussian NLL} = \frac{1}{2}\log(2\pi) + \log(\sigma) + \frac{(z-\mu)^2}{2\sigma^2}$$
其中 $\log(\sigma)$ 项防止模型无限放大 $\sigma$ 来逃避 MSE 项惩罚。

---

### Q3: Teacher Forcing 是什么？训练和推理有什么区别？会导致什么问题？

**一句话**：训练时每步输入真实值（Teacher Forcing），推理时每步输入自己的采样值，这种不一致叫做 **exposure bias**。

**展开**：
- **训练**：$t$ 时刻 LSTM 输入 $z_{t-1}^{real}$（上一步的实际观测值），相当于有"标准答案"引导。
- **推理**：$t$ 时刻 LSTM 输入 $\hat{z}_{t-1}$（上一步自己采样出的值），一旦采样偏了，后续预测会**逐步漂移**。
- 这导致推理时的预测分布和训练时的条件分布存在 gap，预测越长越明显。

**如何缓解**：
- **Scheduled Sampling**：训练时以一定概率用模型自己的输出替代真实值，让模型适应自己的错误。
- **增加采样路径数 K**：K 足够大（200~1000）时分位数更稳健。
- **缩短 context_length**：减少对远期历史的依赖。

**加分**：Exposure bias 是自回归模型（AR RNN, Transformer Decoder）的通病，不仅是 DeepAR 的问题。这也是为什么 Beam Search 在 NLP 中比 Greedy Decoding 更好的原因之一。

---

### Q4: DeepAR 如何做冷启动？Item Embedding 的原理是什么？

**一句话**：通过学到的 item embedding 向量，让模型在没有历史的新序列上也能做合理预测。

**展开**：
- 训练时，每个 item 学到一个稠密向量 $e_i \in \mathbb{R}^{d_e}$，这个向量在每一步和销量值、协变量一起输入 LSTM。
- 相似销量模式的 SKU（如同品类、同价格带）在 embedding 空间会自然靠近。
- 冷启动时，对全新 SKU 可以用**同类已有 SKU 的 embedding 均值**或**从 item 特征（价格、品类、品牌）通过一个小网络映射**得到初始 embedding。

**加分**：可以类比 NLP 中的 word embedding — "SKU_爆款手机壳" 和 "SKU_普通手机壳" 在 embedding 空间接近，所以即使后者是新上架的，模型也能从前者"借"到季节性和趋势模式。

---

### Q5: DeepAR 为什么用 LSTM 而不是 Transformer？

**一句话**：DeepAR 发表于 2017 年（Transformer 刚出现），但 LSTM 在自回归概率预测场景下仍然是合理甚至更优的选择。

**展开对比**：

| 维度 | LSTM (DeepAR) | Transformer (Informer/PatchTST) |
|------|--------------|-------------------------------|
| 推理方式 | 逐步自回归（天然适合） | 一次输出全部（需加因果 mask） |
| 长序列效率 | 慢（$O(T)$ 串行） | 快（$O(1)$ 并行） |
| 概率输出 | 每步采样自然接入 | 需额外的分布头 |
| 冷启动 | Item embedding 直连 | 同样的 embedding 可用 |
| 计算资源 | 轻量 | 内存/显存消耗大 |

**什么时候 LSTM 反而更好**：
- 序列不长（< 200 步），并行优势不明显
- 需要概率输出（Transformer 做概率预测需要额外设计）
- 数据量不够充足（Transformer 更吃数据）

**加分**：现在 GluonTS 里已经有 Transformer-based 的变体（如 Time Series Transformer），实际工程中你可以先跑 DeepAR 做 baseline，再对比 Transformer 类模型。

---

### Q6: Scale Normalization 为什么必要？不做会怎样？

**一句话**：不做 scale 的话，大数值序列的梯度会淹没小数值序列的信号。

**展开**：
- 假设 SKU_A 日销 0~5 件，SKU_B 日销 5000~8000 件。不做 scale，两个序列的原始值相差 3 个数量级。
- LSTM 的激活函数（sigmoid/tanh）在 $[-2, 2]$ 区间梯度最大。5000 这样的大值会让激活饱和，梯度消失。
- Scale normalization（每条序列除以其历史绝对值的均值 $v_i = \frac{1}{t_0}\sum |z_{i,t}|$）把不同量级的序列拉到同一范围，梯度稳定。
- 预测完成后**必须乘回 scale 因子**：$\hat{z}_{orig} = \hat{z}_{normalized} \times v_i$。

**加分**：这是一种**instance normalization** 的思想 — 沿时间维度做归一化，而非沿 batch 维度。更稳健的做法是用中位数而非均值，避免个别极端值影响。

---

### Q7: 祖先采样 (Ancestral Sampling) 和直接取分布均值做预测，有什么区别？

**一句话**：取均值是点预测，祖先采样是分布预测 — 后者保留了不确定性并支持分位数决策。

**展开**：
- **直接取均值** $\hat{z}_t = \mu_t$：快速，但丢失了方差信息。无法回答"销量超过 20 件的概率是多少？"
- **祖先采样**：从 $P(z_t | \mu_t, \sigma_t)$ 随机抽 $K$ 条路径 →
  - 得到 $K$ 条不同的未来轨迹
  - 取分位数 → P10/P50/P90 预测区间
  - 可计算任意事件概率：$P(\sum_{t=1}^7 z_t > \text{库存}) = \frac{\text{count}_{>库存}}{K}$

**为什么采样比直接算分位数更好**：因为 DeepAR 输出的是 $P(z_t | z_{t-1}, ..., h_t)$，这是一个逐步条件分布，联合分布 $P(z_1, ..., z_H)$ 没有闭式解。采样是近似联合分布的标准方法。

**加分**：采样路径数量 $K$ 的选择 — 100 给大致区间，200~500 给稳定分位数，1000+ 给尾部风险（P99）。这本质上是用 Monte Carlo 方法估计联合分布的分位数。

---

### Q8: 如何为你的数据选择合适的似然函数？

**一句话**：看数据分布特征 — 连续选 Gaussian/StudentT，计数选 NegBin，有界选 Beta。

**展开**：

| 你的数据长什么样 | 选这个分布 | 为什么 |
|---|---|---|
| 实数、有正有负、无明显厚尾 | `GaussianOutput()` | 基础选择，μ 预测值，σ 预测不确定性 |
| 实数、有明显异常值/厚尾 | `StudentTOutput()` | 额外参数 ν（自由度）让尾部更厚，不被异常值带偏 |
| 非负整数、方差 > 均值 | `NegativeBinomialOutput()` | 方差 = μ + αμ²，天然适配过离散计数 |
| (0, 1) 区间、如转化率 | `BetaOutput()` | 天然有界，不需要截断 |
| 有限类别、如星级评分 | `CategoricalOutput()` | 多分类，每类一个概率 |

**加分**：你可以通过**残差诊断**验证分布选择 — 预测残差应接近均匀分布（通过 PIT 检验）。如果 PIT 呈 U 型，说明模型过于自信（方差估计偏小），可换 StudentT；如果 PIT 呈倒 U 型，说明方差过大。

---

### Q9: 实际项目中遇到 DeepAR 效果不好，你会从哪些方面排查？

**一句话**：按数据→特征→超参→分布的优先级排查。

**展开**（结构化排查 Checklist）：

```
1. 数据层
   □ Scale normalization 是否做了？每条序列都正确缩放了？
   □ 训练/测试切分是否正确？（不能用未来数据训练）
   □ 序列长度是否足够？（至少 context_length × 3）
   □ 是否存在大量零值/缺失值？（考虑 Zero-Inflated 分布）

2. 特征层
   □ 时间协变量是否充分？（至少包含星期几 + 月份）
   □ Item embedding 维度是否合适？（d_e = 10~50）
   □ 有没有漏掉关键外部特征？（促销、节假日、天气）

3. 超参层
   □ context_length 是否合理？（一般 ≥ prediction_length × 2）
   □ hidden_size 是否过大/过小？（试 32 → 64 → 128）
   □ learning rate 是否合适？（try [1e-3, 1e-2, 5e-2]）
   □ 是否过拟合？（增加 dropout、减少 hidden_size）

4. 分布层
   □ 似然函数是否匹配数据类型？
   □ 试试换 StudentT 替代 Gaussian（很多 case 会改善）
```

**加分**：建议用一个简单的 Seasonal Naive 或 Moving Average 做 baseline，如果 DeepAR 连朴素方法都打不过，优先排查数据问题。

---

### Q10: DeepAR 的 O(n) 复杂度到底是多少？训练和推理的瓶颈分别在哪？

**一句话**：训练 $O(N \times C \times d_h^2)$，推理 $O(K \times H \times d_h^2)$。瓶颈分别是序列长度 $C$ 和采样数 $K$。

**展开**：

- **训练复杂度**：一个 batch 的 LSTM 前向 + 反向 $O(B \times C \times d_h^2)$，$C$ 是 context_length。LSTM 无法并行化时间维度，所以 $C$ 越长训练越慢。
- **推理复杂度**：对一条序列 $O(K \times H \times d_h^2)$，$K$ 是采样路径数。每条路径独立逐步展开，无法并行。1000 条序列 × 200 条采样 = 200,000 次 LSTM unrolling，这是主要耗时来源。

**优化思路**：
- 训练：减小 `context_length`（如从 200 → 56），batch 级别 GPU 并行
- 推理：采样路径可以 batch 化（构造 `(B×K, ...)` 的输入，一次 LSTM 前向处理多条路径）
- 如果延迟是关键（如在线预测），可用模型蒸馏或 ONNX 导出加速

**加分**：GluonTS 的预测器内部已经做了采样路径的 batch 化优化，不需要手动处理。

---

### Q11: 为什么 DeepAR 对长预测窗口 (>100) 表现变差？

**一句话**：因为祖先采样的**误差累积** — 每一步的采样误差都会传导到下一步，$t=100$ 时方差已被放大到一个无法控制的程度。

**展开**：
- 这是所有自回归生成模型的固有问题（NLP 的文本生成长度也有限制）。
- 数学上：$\text{Var}(z_{t+H}) \approx \sigma^2 \times H$（简化假设），方差随预测长度近似线性增长。
- 直观理解：预测明天天气有 70% 准确率，预测 30 天后天气几乎随机。

**应对策略**：
- 增大 `context_length` 提供更多历史信息
- 使用多步 ahead 损失（不只优化的 one-step NLL）
- 换用非自回归模型（如 N-BEATS 直接输出整个预测窗口，但失去概率输出）
- 使用 Transformer 类模型利用长距离依赖

**加分**：实际使用中，DeepAR 的**分位数预测区间**虽然宽度随 H 增大而变宽（这本身是合理的），但**中位数点预测**在 H > 100 时是否退化取决于数据中是否存在足够长的周期模式。

---

### Q12: 面试中可能被问到的对比题

> 以下给出"一句话定位"——

| 对比 | 关键区别 |
|------|---------|
| **DeepAR vs N-BEATS** | DeepAR 是概率自回归（RNN），N-BEATS 是点预测前馈（FC stack）。前者给分布，后者给点。 |
| **DeepAR vs DeepState** | 同门师兄弟。DeepState 多了线性状态空间层，对趋势/季节有结构化假设；DeepAR 完全让 LSTM 自学习。 |
| **DeepAR vs Prophet** | Prophet 是加法分解（trend + season + holiday），强可解释但弱于非线性模式；DeepAR 黑盒但拟合能力强。 |
| **DeepAR vs LightGBM (你的老技能栈)** | GBDT 需要手工特征工程（lag, rolling mean, ...），DeepAR 自动学时间依赖。但 GBDT 训练快、可解释性强、数据少时也能用。 |
| **DeepAR vs PatchTST** | PatchTST 用 Transformer 分 patch 建模长序列，效率高但输出是点预测。可以在 PatchTST 上加分布头变成概率模型。 |

**加分金句**："DeepAR 的价值不在 LSTM 本身（LSTM 已不是最新架构），而在于它开创了**全局概率自回归**这一范式的先河。后来的 DeepState、MQ-RNN、甚至时序大模型，都受这个思路影响。"

---

### Q13: 如果面试官让你现场画 DeepAR 架构图

推荐画出以下 5 个模块（从输入到输出，标注数据流向）：

```
   ┌──────────────┐
   │ z_{t-1}      │  上一步销量 (标量)
   └──────┬───────┘
          │
   ┌──────┼───────┐
   │ cov_t │ emb_i │  协变量 + item embedding
   └──────┼───────┘
          │
   ┌──────▼───────┐
   │   CONCAT     │  → 向量 [z, cov, emb]  (1 + d_c + d_e 维)
   └──────┬───────┘
          │
   ┌──────▼───────┐
   │    LSTM      │  → h_t  (d_h 维隐状态)
   │  (2 layers)  │
   └──────┬───────┘
          │
   ┌──────▼───────┐
   │  Linear +    │  → μ_t, σ_t (或 μ_t, α_t)
   │  softplus    │
   └──────┬───────┘
          │
   ┌──────▼───────┐
   │ Loss: -log P │  ← 训练: z_t^{real}  推理: z_t ~ P(z|μ,σ)
   └──────────────┘
```

**边画边说的要点**：
1. "输入是三个东西的拼接 — 上一步值、协变量、序列 embedding"
2. "LSTM 输出隐状态，线性层映射到两个分布参数"
3. "训练用真实值算 NLL，推理从分布采样再输回 LSTM（箭头往回画一下）"

---

> **面试总结**：DeepAR 的面试考察点集中在 **(1) 概率预测 vs 点预测的认知 (2) Teacher Forcing / 祖先采样的机制 (3) 全局建模 vs 局部建模的 tradeoff (4) 工程落地细节（scale, embedding, 分布选择）**。掌握这四块，基本覆盖 90% 的问题。
