# iTransformer: 把 Transformer 倒过来——变量维度的注意力

> **论文**: [iTransformer: Inverted Transformers Are Effective for Time Series Forecasting](https://arxiv.org/abs/2310.06625)
> **机构**: 清华大学 (ICLR 2024)
> **作者**: Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, Mingsheng Long

---

## 1. 核心思想

iTransformer 的核心洞察：**传统 Transformer 在时间序列上把"时间步"当 token，这是照搬 NLP 的经验主义——时间序列的变量（variates）往往是不同物理量（温度、湿度、风速），它们之间的关系比远距离时间步之间的关系更有建模价值。**

一句话概括：**把每个变量的整条历史序列嵌入为一个 token，在变量维度上做 Self-Attention 捕捉跨变量依赖，在时间维度上用 FFN 捕捉时序模式。简单说——把 Transformer 倒过来用。**

```
标准 Transformer: 时间步→token, Attention(时间步, 时间步), FFN 共享
iTransformer:     变量→token,  Attention(变量, 变量),    FFN 独享
```

举例：预测未来 96 步的温度，输入有温度、湿度、气压、风速 4 个变量，回溯 96 步。
- **标准做法**：96 个 token，每个是 4 维向量，注意力在 96 个时间步之间
- **iTransformer**：4 个 token，每个是 96 维向量（一个变量的完整历史），注意力在 4 个变量之间

标准做法的问题是：时间步 token 信息量少（单个时刻），且单变量预测时多变量被混成多维特征，主变量信号被淹没。iTransformer 的倒置让 token 信息量变丰富（整个历史），同时注意力自然地关注"哪个变量对谁有影响"。

---

## 2. 演进路线（Transformer 家族在时序预测中的演进）

```
Transformer (2017, NLP) ── "Attention Is All You Need"
        │
        ├─→ Informer (AAAI 2021) ── 解决 O(L²) 复杂度问题 (ProbSparse Attention)
        ├─→ Autoformer (NeurIPS 2021) ── 自相关替代点积注意力 + 渐进分解
        ├─→ FEDformer (ICML 2022) ── 频域增强 + 随机傅里叶
        │
        ├─→ PatchTST (ICLR 2023) ──★ 关键转折：把时间序列切成 patch，每个 patch 当 token
        │                              Channel-Independent：每个变量独立建模
        │
        ├─→ Crossformer (ICLR 2023) ── 两阶段注意力：时间 patch + 跨变量 patch
        │
        ├─→ iTransformer (ICLR 2024) ──★★ 本文：倒置！变量→token，时间→FFN
        │
        ├─→ TimeXer (NeurIPS 2024) ── 双粒度：内生变量 patch-level + 外生变量 series-level
        │
        └─→ PatchTrase (?) ── patch + trace 混合，继续在"维度倒置"思路上的迭代
```

**关键洞察链**：
1. **PatchTST** 发现：单变量独立建模（Channel-Independent）反而比多变量混合效果好 → 说明变量间耦合方式需要重新思考
2. **iTransformer** 进一步推导：如果变量信息不能混在时间轴 token 里，不如反过来——每个变量当作独立 token，注意力专门学习变量间关系
3. 这条线最终导向：**多变量建模的问题不在"要不要建模变量间关系"，而在"怎么建模"——把变量关系交给 Attention，把时序模式交给 FFN**

---

## 3. 模型架构

### 3.1 整体公式

给定输入 $\mathbf{X} \in \mathbb{R}^{L \times C}$（$L$=回溯长度，$C$=变量数）：

$$\mathbf{X}^0 = \text{Embed}(\mathbf{X}^\top) \quad \in \mathbb{R}^{C \times d_{\text{model}}}$$

$$\mathbf{X}^{l+1} = \text{iTransformerBlock}^l(\mathbf{X}^l)$$

$$\hat{\mathbf{Y}} = \text{Projector}(\mathbf{X}^{\text{final}}) \quad \in \mathbb{R}^{C \times H}$$

其中 $\mathbf{X}^\top$ 把 L×C 转置为 C×L —— 每个变量成为一行，长度为 L。Embed 层把 L 维投影到 $d_{\text{model}}$ 维。

### 3.2 与传统 Transformer 的对照

```
传统 Transformer (时间步=token)            iTransformer (变量=token)
═══════════════════════════════            ═════════════════════════
X ∈ R^{L×C}                               X ∈ R^{L×C}
   ↓ 每个时间步 embed                         ↓ 转置
tokens ∈ R^{L×d} (L 个 token)              X^T ∈ R^{C×L}
   ↓                                          ↓ 每个变量 embed
Self-Attn 在 L 个 token 间               tokens ∈ R^{C×d} (C 个 token)
   复杂度: O(L²·d)                           ↓
   ↓                                      Self-Attn 在 C 个变量间
FFN 逐时间步（共享权重）                     复杂度: O(C²·d)
   每个时间步的 d 维向量 → d 维               ↓
   ↓                                      FFN 逐变量（共享权重）
输出: R^{L×d}                               每个变量的 d 维向量 → H 维
   ↓ 投影                                     ↓ 投影
Y ∈ R^{H×C}                                Y^T ∈ R^{C×H} → Y ∈ R^{H×C}

核心差异:
• token 数量: L → C（通常 C << L，更高效）
• Attention 语义: "t1 和 t2 有什么关系" → "变量 i 和变量 j 有什么关系"
• FFN 语义:    "这一步特征怎么变换" → "这个变量的时序模式怎么外推"
• 复杂度:     O(L²) → O(C²)（对长序列友好）
```

### 3.3 倒置嵌入（Inverted Embedding）

传统 Transformer 将每步的 C 维向量 → d_model 维。iTransformer 将每个变量的 L 维向量 → d_model 维：

$$\text{Token}_i = \text{Embed}(\mathbf{X}_{:,i}) \quad i=1,\dots,C$$

Embed 是一个线性投影：`nn.Linear(L, d_model)`。还可叠加可学习的位置编码（变量维度的位置编码，指示"这是第几个变量"）和变量特定嵌入。

```python
# 伪代码：倒置嵌入
X: (B, L, C)
X = X.permute(0, 2, 1)           # (B, C, L)  — 每个变量一条 L 维向量
tokens = self.embed(X)            # (B, C, d_model) — C 个变量 token
```

### 3.4 iTransformer Block

每个 block 包含两个子层，结构与标准 Transformer Encoder 一致，但操作对象不同：

```
┌─────────────────────────────────────┐
│ LayerNorm  ─── 沿变量维度(C)归一化     │
│     ↓                               │
│ Multi-Head Self-Attention           │
│   • Q,K,V 来自 C 个变量 token       │
│   • 学习第 i 个变量与第 j 个变量的关系 │
│     ↓                               │
│ + Residual                          │
└─────────────────────────────────────┘
│
▼
┌─────────────────────────────────────┐
│ LayerNorm  ─── 沿变量维度(C)归一化     │
│     ↓                               │
│ Feed-Forward Network (共享权重)       │
│   每个变量 token 独立通过 FFN：       │
│    d_model → d_ff → d_model        │
│   FFN 学习：从历史模式映射到未来模式   │
│     ↓                               │
│ + Residual                          │
└─────────────────────────────────────┘
```

**关键理解**：
- **Attention** 跨变量操作——"温度对湿度有什么影响？"
- **FFN** 在时间维度上操作——"这个变量的历史规律如何延伸到未来？"
- **LayerNorm** 沿变量维度——对 C 个 token 做归一化（对应标准 Transformer 沿 token 维度做 LN 的对称操作）

### 3.5 输出投影

最后一个 block 的输出 $\mathbf{X}^{\text{final}} \in \mathbb{R}^{C \times d_{\text{model}}}$，每个变量一个 d_model 维 token。投影到预测长度 $H$：

$$\hat{\mathbf{Y}} = \text{Projector}(\mathbf{X}^{\text{final}}) \in \mathbb{R}^{C \times H}$$

Projector 通常是 `nn.Linear(d_model, H)`。最后转置回 $(B, H, C)$ 与标签对齐。

### 3.6 数据流概览

以天气预测为例：回溯 96 步（$L=96$），4 个变量温度/湿度/气压/风速（$C=4$），预测未来 96 步（$H=96$），$d_{\text{model}}=512$。

**训练**：
1. 输入 `(B, 96, 4)` → 转置 `(B, 4, 96)` → Embed `(B, 4, 512)` — 4 个 token
2. N 层 iTransformer Block → Attention `(4×4)` 学习变量间关系；FFN 每个变量独立转换
3. 输出 `(B, 4, 512)` → Projector `(B, 4, 96)` → 转置 `(B, 96, 4)` → MSE Loss

**推理**：单步前向传播（非自回归），直接输出 $(H, C)$ 预测。这一点与 DeepAR（逐步自回归）和 TCN（并行卷积）都不同——iTransformer 完全是非自回归的。

> 完整的数据流转和每一步 shape 变化见 `code/04_transformer/03_itransformer_demo.py`。

---

## 4. 关键设计决策

### 4.1 为什么变量当 token 比时间步当 token 好？

| 维度 | 时间步=token（传统） | 变量=token（iTransformer） |
|------|---------------------|--------------------------|
| Token 信息量 | 单个时刻的 C 维快照（信息稀疏） | 整条 L 维历史（信息稠密） |
| Attention 语义 | "t 时刻和 t+k 时刻有关吗？" | "变量 A 对变量 B 有因果影响吗？" |
| 时间依赖建模 | Attention（每个头都是一套全局依赖） | FFN（每个变量学自己的时序外推） |
| 多变量耦合 | 混在一个 token 里，不显式建模 | Attention 专门学习变量间关系 |
| 复杂度 | $O(L^2)$ | $O(C^2)$，通常 $C \ll L$ |
| 长序列友好 | 差（平方增长） | 好（与 L 无关） |

核心论据：**时间序列多变量预测中，时间依赖的模式相对稳定（FFN 就能学会），而变量间的相互影响才是需要 Attention 捕捉的结构化知识。**

### 4.2 FFN 为什么能学时序模式？

这可能是 iTransformer 最容易被质疑的点："FFN 是按位置独立操作的，你怎么让它学时间依赖？"

答案在于**嵌入层的角色**。每个变量的 token 从 L 维投影到 d_model 维时，线性映射 `W ∈ R^(L×d)` 的每一列就是一个**基滤波器**。token 的每个维度编码了该变量在时间轴上的某种 pattern。FFN 在 d_model 维度上做非线性组合，本质上是在组合不同的时序模式。

> 类比：DeepAR 的 LSTM 隐状态也是一个固定维度向量——它通过循环逐步积累时间信息。iTransformer 的 token 是整条序列一次性投影进固定维度——它通过线性基展开来编码时间信息。一个是递推，一个是投影，但目标相同：把变长序列压缩为定长表示。

### 4.3 变量维度 LayerNorm 的含义

标准 Transformer：LayerNorm 沿 token 维度（NLP 中是序列长度方向），每个 token 的特征做归一化。
iTransformer：LayerNorm 沿变量维度（C 个 token），每个 token 的特征做归一化。

**物理含义**：把每个变量的表示归一化到同一尺度——这样 Attention 计算的变量间相似度不受各变量绝对量级的影响。不做这个归一化，数值范围大的变量（如气压 ~1000）会主导对数值范围小的变量（如湿度 ~0.5）的关注。

---

## 5. 训练与推理

### 5.1 训练

- **非自回归**：一次前向给出全部 H 步预测
- **Loss**：MSE（论文默认）或 MAE
- **与 DeepAR 的关键区别**：不需要 Teacher Forcing，不需要 Ancestral Sampling
- **输入**：$(L, C)$ 回溯窗口 → **输出**：$(H, C)$ 多步预测

### 5.2 推理

- 纯前向传播，无迭代采样
- 速度与输入长度 L 线性相关（Transformer 部分与 L 无关，Embed 部分是线性投影 $O(L·d)$）
- 对比 DeepAR（逐步 $O(H)$ 次 LSTM 前向），iTransformer 在长预测步长场景更快

---

## 6. 优缺点

### 优点

1. **多变量建模优雅**：Attention 专门学变量关系，FFN 专门学时间模式，职责分离
2. **长序列高效**：Attention 复杂度 $O(C^2)$ 与 L 无关，可处理很长的回溯窗口
3. **变量数扩展性好**：$C$ 增加到数百时仍可控（相比 $L$ 通常成百上千）
4. **非自回归**：一次前向出全部预测，无误差累积
5. **实现简单**：只需把标准 Transformer 的输入转置 + embedding 维度改一下

### 缺点

1. **$C$ 很大时吃不消**：$O(C^2)$ 注意力在高维传感器场景（$C > 1000$）成为瓶颈
2. **$C=1$ 时退化**：单变量场景只有一个 token，Attention 毫无意义，退化为"Embed + FFN"——这是一个线性+MLP 模型，效果不一定好
3. **不支持概率预测**：原生是点预测（MSE），要做概率预测需要修改输出头（Gaussian/Quantile）
4. **Embed 层是瓶颈**：$L \times d_{\text{model}}$ 的线性投影在大 L 和大 d_model 下参数量大（如 $L=512$, $d=512$ → 26 万参数仅 Embed 层）
5. **时间信息压缩有损**：整条 L 维序列压成一个 d_model 维向量，长序列的细节可能丢失

### 单变量退化分析

$C=1$ 时 iTransformer 的结构退化为：

```
单变量序列 → Embed(L→d) → [无 Attention，因为只有 1 个 token] → FFN → Projector(d→H)
```

这就是一个两层的 MLP + 残差结构，跟 DLinear（一层 Linear）比没有显著优势，跟 PatchTST（还保留 patch-level attention）比更是劣势。所以**只在 $C \geq 3$ 的多变量场景考虑 iTransformer**。

---

## 7. 2026 定位

```
2023 前的 Transformer 时序模型 ── 都在优化时间维度的 Attention（稀疏、低秩、频域...）
               │
iTransformer (2024) ── 换了一个维度做 Attention，被 ICLR 接收
               │
后续发展:
  ├─ TimeXer (NeurIPS 2024) ── 双粒度注意力，把 iTransformer 的"变量级"和 PatchTST 的"patch 级"融合
  ├─ iTransformer 思想影响 ── 后续多变量模型普遍考虑"时间/变量分离建模"
  └─ 2026 现状 ── 已成为多变量时序预测的标准 baseline 之一
```

到 2026 年，iTransformer 已经不是一个"新模型"了，而是一个**稳定的 baseline**——在多变量场景用它跑一版结果，拿不下再用更复杂的方法（TimeXer、PatchTST+Cross-Attention）。它的主要价值在于：
- **作为强 baseline** 验证新方法的有效性
- **变量数量 3~50 的多变量预测** 使用
- **学术对比** 中的标准方法之一

单变量场景首选 PatchTST 或 DLinear，多变量（C 适中）选 iTransformer，大量变量（C>100）+ 长序列选现代 SSM（Mamba）。

---

## 8. 关键超参数

| 参数 | 建议范围 | 作用 |
|------|---------|------|
| d_model | 256~512 | 变量 token 的表示维度（论文默认 512） |
| n_heads | 8~16 | 多变量关系建模的多头数 |
| e_layers | 2~4 | iTransformer Block 层数 |
| d_ff | 512~2048 | FFN 隐层维度 |
| dropout | 0.1 | 防止过拟合 |
| L (lookback) | 96~336 | 回溯窗口长度 |

**调参优先级**：$L$（回溯长度）> d_model > e_layers > d_ff。

$L$ 是最关键的超参数——太短则 token 信息不足，太长则 Embed 参数量过大。建议从 $L=H$（回溯=预测长度）起步。

---

## 9. 实践建议

1. **先评估是否需要多变量建模**：如果各个变量之间确实有物理上的因果关系（温度→用电量、汇率→股价），iTransformer 的跨变量注意力有价值。如果变量间独立或弱相关（不同门店销量），Channel-Independent（PatchTST 方式）可能更好。

2. **$C$ 的选择**：不要把所有可用变量都扔进去。做特征选择，保留与被预测变量有 Granger 因果或其他关联的变量。$C$ 越小，注意力越集中，噪声越少。

3. **归一化要慎重**：每个变量独立做 z-score 归一化（RevIN 推荐方式），避免一个变量碾压其他变量。iTransformer 论文使用 RevIN（可逆实例归一化）作为标配。

4. **与 PatchTST 的选择**：
   - 多变量有明显交互 → iTransformer
   - 多变量但相互独立 → PatchTST（Channel-Independent）
   - 不确定时 → 两个都跑，选验证集更好的

5. **Embed 层参数量控制**：$L \times d_{\text{model}}$ 可能很大。如果 $L=336$ 且 $d=512$，Embed 层有 17 万参数。可以用两阶段投影降参：`nn.Sequential(nn.Linear(L, d//2), nn.ReLU(), nn.Linear(d//2, d))`——参数量从 $L·d$ 降到 $L·(d/2) + (d/2)·d$。

---

## 10. 生产级工具链

### 官方仓库（推荐方式）

```bash
git clone https://github.com/thuml/iTransformer.git
```

iTransformer 官方代码属于 [Time-Series-Library (TSLib)](https://github.com/thuml/Time-Series-Library) 生态——清华大学软件学院维护的统一时序预测框架，在同一 `experiments/` 下可切换 Informer/Autoformer/FEDformer/PatchTST/iTransformer/TimeXer 等十多种模型。

### 使用方式

```python
# 官方 TSLib 风格（简化示例）
from models.iTransformer import Model

model = Model(
    configs=dict(
        enc_in=7,        # 变量数 C
        seq_len=96,      # 回溯长度 L
        pred_len=96,     # 预测长度 H
        d_model=512,
        n_heads=8,
        e_layers=3,
        d_ff=2048,
        dropout=0.1,
        use_norm=True,   # RevIN
    )
)
```

### 第三方集成

- **tsai** (`pip install tsai`): `from tsai.models.iTransformer import iTransformer` — 社区实现
- **neuralforecast** (`pip install neuralforecast`): 截至 2025 尚未原生支持，可通过自定义模型接口集成
- **PyTorch Forecasting**: 截至 2025 尚未原生支持

> 本笔记配套代码 `code/04_transformer/03_itransformer_demo.py` 提供了独立可运行的 PyTorch 实现，不依赖 TSLib。

---

## 11. 进一步阅读

- [iTransformer 论文](https://arxiv.org/abs/2310.06625)
- [官方代码 (Time-Series-Library)](https://github.com/thuml/Time-Series-Library)
- [PatchTST 论文](https://arxiv.org/abs/2211.14730) — 理解 Channel-Independent 的设计哲学
- [TimeXer 论文](https://arxiv.org/abs/2402.19072) — iTransformer 思想的延续：双粒度变量建模
- [Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504) — DLinear 论文，理解为什么简单方法能打败复杂 Transformer

---

## 12. 面试问题

**Q1: iTransformer 的核心创新是什么？用一句话概括。**

> 把 Transformer 的维度倒过来——变量当 token（Attention 学变量间关系），时间维度交给 FFN（学时序模式）。标准 Transformer 是"时间步之间做 Attention"，iTransformer 是"变量之间做 Attention"。

**Q2: 为什么"倒过来"更合理？**

> 两个理由：(1) 变量 token 信息量更丰富——每个 token 包含变量的完整 L 步历史，而不是单个时间步的 C 维快照；(2) 多变量预测的核心挑战是变量间耦合关系，Attention 天然适合捕捉这种依赖，而时序外推这种"单调操作"FFN 就能完成。

**Q3: iTransformer 的 FFN 是怎么学习时序模式的？**

> 嵌入层 `Linear(L, d_model)` 将 L 步历史投影到 d_model 维，权重矩阵的每一列是一个可学习的时序基滤波器。每个 token 的 d_model 个维度编码了不同的时序 pattern。FFN 在 d_model 维度上做非线性变换，组合不同的时序模式来预测未来。

**Q4: 为什么 iTransformer 的 LayerNorm 沿变量维度而不是时间维度？**

> 因为 iTransformer 的 token 是变量而不是时间步。沿变量维度做 LayerNorm = 把每个变量的表示归一化到相同尺度，避免数值大的变量（如气压 ~1000）在 Attention 中主导对小变量（如湿度 ~0.5）的关注。这与标准 Transformer 沿 token 维度做 LN 的动机一致。

**Q5: 单变量场景用 iTransformer 合适吗？**

> 不合适。$C=1$ 时只有一个 token，Attention 退化为单位映射。模型退化为一维 Embed + FFN + 残差的 MLP，效果不如 DLinear 或 PatchTST。iTransformer 的设计前提是多变量（$C \geq 3$ 比较合适）。

**Q6: iTransformer vs PatchTST 怎么选？**

> - 变量间有明显耦合关系（因果关系、物理约束） → iTransformer：Attention 显式建模变量关系
> - 变量间相互独立或弱相关 → PatchTST（Channel-Independent）：每个变量单独建模，避免无关变量引入噪声
> - 需要在大量变量上做预测（$C > 100$）且变量独立 → PatchTST
> - 需要在中等数量变量上做预测（$3 \leq C \leq 50$）且有交互 → iTransformer

**Q7: iTransformer 的复杂度是多少？**

> Self-Attention: $O(C^2 \cdot d)$（$C$ = 变量数），与回溯长度 $L$ 无关。
> FFN: $O(C \cdot d \cdot d_{ff})$。
> Embed: $O(C \cdot L \cdot d)$。
> 总体与 $L$ 线性相关、与 $C$ 平方相关。适合"长回溯、少变量"场景。

**Q8: iTransformer 能做概率预测吗？**

> 原生论文是点预测（MSE Loss）。要扩展为概率预测，可以改输出头为分布参数（如 Gaussian 的 μ, σ），但论文未做这个方向。如果需要概率预测，DeepAR 或基于分位数回归的模型更合适。

---

## 13. 与其他模型的对比

| 维度 | DeepAR | TCN | DLinear | **iTransformer** |
|------|--------|-----|---------|-----------------|
| 核心机制 | 自回归 RNN | 因果膨胀卷积 | 趋势分解+线性 | 倒置 Attention |
| 时间建模 | LSTM 循环 | 卷积核扫描 | 线性矩阵乘法 | FFN 逐变量 |
| 变量建模 | 嵌入拼接 | 通道维卷积 | 逐通道独立 Linear | Attention 跨变量 |
| 概率预测 | ✅ 原生支持 | ❌ 点预测 | ❌ 点预测 | ❌ 点预测（可扩展） |
| 预测方式 | 自回归（逐步） | 并行卷积 | 并行 Linear | 并行（非自回归） |
| 长序列效率 | 差（逐步 $O(H)$） | 好（膨胀卷积） | 极好（矩阵乘） | 好（复杂度与 L 无关） |
| 多变量支持 | 通过 Embedding | 通过通道维 | Channel-Independent | ✅ 原生设计 |
| 单变量表现 | 好 | 好 | 好 | 差（退化） |
| 2026 定位 | 概率预测首选 | CNN baseline | 极简 baseline | 多变量 baseline |
