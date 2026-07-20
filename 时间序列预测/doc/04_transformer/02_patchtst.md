# PatchTST: 时间序列即 patch——Transformer 时序预测的新范式

> **论文**: [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://arxiv.org/abs/2211.14730)
> **机构**: UC Berkeley & ServiceNow Research (ICLR 2023)
> **作者**: Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, Jayant Kalagnanam

---

## 1. 核心思想

PatchTST 回答了一个关键问题：**Transformer 处理时间序列时，真的需要把每个时间步当作一个 token 吗？**

答案来自一个看似简单但极其重要的观察：**单个时间步包含的信息太少，就像自然语言中单个字母没有语义一样——应该把连续的多个时间步打包成一个 patch。**

### 从 NLP 的类比开始

NLP 领域（Transformer 的发源地）：
```
句子: "I love machine learning"
  → Token: ["I", "love", "machine", "learning"]  ✓ 有语义
  → 不拆成: ["I", " ", "l", "o", "v", "e", ...] ✗ 无意义
```

时序领域（Transformer 被照搬使用）：
```
序列: [23.1, 23.5, 24.0, 24.8, 25.1, 25.6, ...]

旧做法（逐点token）:         PatchTST（patch token）:
Token1: [23.1]  ← 没语义    Patch1: [23.1, 23.5, 24.0, 24.8]
Token2: [23.5]  ← 没语义    Patch2: [25.1, 25.6, 26.0, 26.3]
Token3: [24.0]  ← 没语义    Patch3: [26.8, 27.0, 27.2, 26.9]
```

**关键洞察**：单个时间步的数字（如 23.1°C）本身没有"含义"，只有连续的一段才形成"趋势"或"模式"——就像只有连续的字母才形成单词。

### 三个核心创新

**创新一：Patching（时序切片）**

把长序列切成 patch，每个 patch 是 P 个连续时间步。三重好处：
1. **降低序列长度**：96 个时间步 → 96/P 个 patch → Attention 复杂度从 O(96²) 降到 O((96/P)²)
2. **保留局部语义**：每个 patch 是一小段波形，有明确的"形状"（上升、下降、波动）
3. **大幅降低显存**：token 数量减少 = 注意力矩阵减小

**创新二：Channel-Independent（变量独立建模）**

每个变量独立建模，共享编码器但分别预测。这与当时的主流想法相反——大家都认为"建模变量间关系"很重要。但 PatchTST 实验发现：让模型专注于单变量的时序模式，反而比变量混在一起效果更好。变量间关系丢失一些信息，但 Patch 带来的时序建模能力提升远超损失。

**创新三：Self-Supervised Pre-training（自监督预训练）**

借鉴 NLP 的 BERT，对时序做 Masked Patch Modeling：随机遮盖部分 patch，从周围 patch 重建被遮盖部分。在数据不足时尤其有效。

### 一句话总结

**把时间序列切成 patch（像 NLP 的 token）+ 每个变量独立建模 + 可选的预训练 = 简单而极其有效的长序列预测。**

```
标准 Transformer:     时间步→token  →  O(L²) Attention + 变量混合
Informer/Autoformer:  降低复杂度    →  优化"怎么算注意力"
PatchTST:             patch→token   →  改变"什么是 token"
```

---

## 2. 演进路线：Patch 方法论的兴起

```
Transformer (2017) — 时序: token=时间步
    │  问题: 单个时间步语义贫乏，序列太长
    │
    ├─→ Informer (2021) — ProbSparse Attention
    ├─→ Autoformer (2021) — 时延自相关
    ├─→ FEDformer (2022) — 频域增强
    │   以上都在优化"如何计算注意力"
    │
    ├─→ ★ PatchTST (ICLR 2023) — 改变 token 定义
    │   不优化注意力本身，而是改变输入表示
    │   贡献: Patching + Channel-Independent + 预训练
    │
    ├─→ TimesNet (ICLR 2023) — Patch + 多周期分解
    ├─→ Crossformer (ICLR 2023) — Patch + 跨变量交互
    └─→ iTransformer (ICLR 2024) — 倒置：变量→token
```

**PatchTST 的历史地位**：2023 年之前论文在优化"如何计算注意力"，PatchTST 之后转向"什么应该成为 token"。它开创了"Patch+X"范式，证明简单的 patch 操作胜过复杂的注意力设计。

---

## 3. 模型架构

### 3.1 整体结构

```
输入: X ∈ R^{B×L×C}

        ┌──────────────────────────────┐
        │  Channel-Independent: 每个变量独立处理  │
        └──────────────────────────────┘

变量 1 (L步)      变量 2            ...      变量 C
    │                │                          │
    ▼                ▼                          ▼
┌────────┐    ┌────────┐              ┌────────┐
│Patching│    │Patching│              │Patching│
│L→N patch   │L→N patch              │L→N patch
└────────┘    └────────┘              └────────┘
    │                │                          │
    ▼                ▼                          ▼
┌────────┐    ┌────────┐              ┌────────┐
│Patch   │    │Patch   │              │Patch   │
│Embedding   │Embedding              │Embedding
│P维→d维  │    │P维→d维  │              │P维→d维  │
└────────┘    └────────┘              └────────┘
    │                │                          │
    ▼                ▼                          ▼
┌──────────────────────────────────────────────┐
│     Transformer Encoder (所有变量共享权重)      │
│     Self-Attention 在 N_patch 个 token 之间    │
└──────────────────────────────────────────────┘
    │                │                          │
    ▼                ▼                          ▼
┌────────┐    ┌────────┐              ┌────────┐
│Flatten │    │Flatten │              │Flatten │
│+Linear │    │+Linear │              │+Linear │
│→H步预测│    │→H步预测│              │→H步预测│
└────────┘    └────────┘              └────────┘
    └────────────────┴──────────────────────────┘
                     │
                     ▼
            Ŷ ∈ R^{B×H×C}
```

要点：Encoder 共享（时序模式有通用性），每个变量单独通过但权重相同，预测头独立（外推方式可能不同）。

### 3.2 Patching：核心操作

#### 为什么 Patch 比逐点更好？

**直觉1：语义密度**
```
单点: [23.1] → 看不出趋势
Patch(P=8): [23.1, 23.5, 24.0, 24.8, 25.1, 25.6, 26.0, 26.3] → 上升趋势！
```

**直觉2：利用局部相关性**

相邻时间步高度相关，patch 天然将相关点组合——同一 patch 内共享局部模式。

**直觉3：大幅降维**
```
L=96, P=16: 96个token → 6个token。Attention复杂度降256倍！
```

#### Patch 切分过程

```python
# x: (B, L, C) → (B, C, N_patch, P)
x = x.permute(0, 2, 1)                    # (B, C, L)
x = x.unfold(dimension=-1, size=P, step=stride)  # (B, C, N, P)
```

**stride 选择**：
- `stride = P`（不重叠，论文推荐）：L=96, P=16 → 6 个 patch，简单够用
- `stride = P/2`（重叠一半）：更多 token，更平滑，但计算量增加

#### Patch Embedding

```python
class PatchEmbedding(nn.Module):
    def __init__(self, patch_len, d_model):
        super().__init__()
        self.projection = nn.Linear(patch_len, d_model)
    
    def forward(self, x):
        """x: (B, C, N_patch, P) → (B, C, N_patch, d_model)"""
        return self.projection(x)  # 每个 patch (P维) → (d维)
```

**位置编码**：`pos_encoding[0]`=第1个patch时间段, `pos_encoding[1]`=第2个patch时间段...

### 3.3 Channel-Independent：变量独立建模

#### 两种模式对比

**Channel-Mixing（传统）**：所有变量的所有 patch 混在一起做 Attention
**Channel-Independent（PatchTST）**：每个变量单独通过 Encoder

#### 为什么独立更好？

**1. 避免变量间噪声干扰**：温度预测时，湿度的随机波动不会误导温度
**2. 不同变量的时序模式差异大**：温度有日周期，气压变化缓慢——混一起互相干扰
**3. 工程灵活**：变量数 5 或 862（Traffic），架构不变

#### 消融实验

| 配置 | ETTh1 MSE | 显存 |
|------|----------|------|
| Channel-Mixing | 0.438 | 高 |
| **Channel-Independent** | **0.413** | 低 (~1/C) |

#### 共享策略

```python
# 方案1: Encoder共享 + Head独立（论文默认）
encoder = TransformerEncoder(...)                      # 1个
heads = nn.ModuleList([LinearHead() for _ in range(C)]) # C个

# 方案2: 全部共享（最省显存）
# 方案3: 全部独立（变量间差异极大时用）
```

### 3.4 Transformer Encoder

标准实现，无花哨技巧——Patching 已解决核心瓶颈。

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model, n_heads, d_model*4, dropout, activation='gelu', batch_first=True
)
self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
```

### 3.5 预测头（Flatten + Linear）

```python
class FlattenHead(nn.Module):
    def __init__(self, d_model, n_patches, pred_len):
        super().__init__()
        self.linear = nn.Linear(n_patches * d_model, pred_len)
    
    def forward(self, x):
        """x: (B, N_patch, d_model) → (B, pred_len)"""
        return self.linear(x.reshape(B, -1))  # Flatten + Linear
```

**为什么不用 Decoder？** 非自回归一次性输出全部 H 步，Flatten+Linear 更简单稳定。

### 3.6 完整前向传播

```python
class PatchTST(nn.Module):
    def __init__(self, enc_in=7, seq_len=96, pred_len=96,
                 patch_len=16, stride=8, d_model=128,
                 n_heads=16, n_layers=3, dropout=0.1):
        super().__init__()
        self.n_patches = (seq_len - patch_len) // stride + 1
        
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, self.n_patches, d_model))
        self.dropout = nn.Dropout(dropout)
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, n_heads, d_model*4, dropout,
                activation='gelu', batch_first=True
            ), num_layers=n_layers
        )
        
        self.heads = nn.ModuleList([
            nn.Linear(self.n_patches * d_model, pred_len)
            for _ in range(enc_in)
        ])
    
    def forward(self, x):
        """x: (B, L, C) → (B, H, C)"""
        B, L, C = x.shape
        
        # 1. Patching: (B, L, C) → (B, C, N_patch, P)
        x = x.permute(0, 2, 1)
        x = x.unfold(-1, self.patch_len, self.patch_len)
        
        # 2. Embedding
        x = self.patch_embedding(x) + self.pos_encoding
        
        # 3. 逐变量编码 + 预测
        outputs = []
        for i in range(C):
            x_i = self.dropout(x[:, i])
            x_i = self.encoder(x_i)
            pred_i = self.heads[i](x_i.reshape(B, -1))
            outputs.append(pred_i)
        
        return torch.stack(outputs, dim=-1)
```

---

## 4. 自监督预训练（可选但强大）

### 4.1 核心思想：Masked Patch Modeling

```
NLP (BERT):  "I love [MASK] learning" → 预测 [MASK] = "machine"
时序 (PatchTST): [Patch1] [MASK] [MASK] [Patch4] → 预测 Patch2, Patch3
```

### 4.2 预训练流程

```python
# 阶段1: 预训练（不需要标签）
for x in unlabeled_data:
    # 1. 随机遮盖 40% 的 patch
    mask = random_mask(n_patches, ratio=0.4)
    x_masked = x.clone(); x_masked[:, mask] = 0
    
    # 2. 编码 → 重建被遮盖的 patch
    z = encoder(x_masked)
    pred = reconstruction_head(z[:, mask])
    loss = MSE(pred, x[:, mask])
    loss.backward()

# 阶段2: 微调（正常有监督训练）
model.load_pretrained_weights()
for x, y in labeled_data:
    pred = model(x)
    loss = MSE(pred, y)
    loss.backward()
```

### 4.3 预训练的效果

论文实验（ETTh1，数据减少到 20%）：

| 方法 | 无预训练 MSE | 有预训练 MSE | 提升 |
|------|-------------|-------------|------|
| PatchTST | 0.483 | **0.443** | 8.3% |
| Transformer | 0.532 | 0.510 | 4.1% |

数据少时（20%）预训练增益 8%+，数据充足时（100%）增益 2-3%。PatchTST 受益最大——patch 结构天然适合"遮盖→重建"。

---

## 5. 关键设计决策

### 5.1 Patch 长度选择

论文标题说"64 Words"：L=512, P=16 → 32 个 patch，"64个词就够"是修辞。

| P 值 | Token数 (L=96) | 信息量 | 计算量 | 场景 |
|------|---------------|--------|--------|------|
| P=4 | 24 | 低（太短）| 中 | 高频数据 |
| **P=16** | **6** | **高** | **很低** | **默认推荐** |
| P=32 | 3 | 太高（丢细节）| 极低 | 超长序列 |

**经验公式**：`P ≈ 主周期/3 ~ 主周期/2`
- 小时数据(日周期=24): P=8~12
- 日数据(周周期=7): P=2~4

### 5.2 Channel-Independent 的适用边界

```
✓ 适用：变量间关系弱、变量数多且不稳定、追求训练稳定
✗ 不适用：变量有强因果(动力学系统)、需要理解变量交互
```

### 5.3 预训练：什么时候做？

```
大量标记数据(>10万) → 不需要
中等(1-10万)       → 中等收益(2-5%)
少量(<1万)          → 显著收益(5-15%+)
大量未标记数据      → 强烈推荐
```

---

## 6. 训练与推理

- **训练**：非自回归一次输出 H 步，MSE loss，lr=1e-4 + ReduceLROnPlateau
- **推理**：单次前向，复杂度 O(C·(L/P)²·d)
- **加速比** vs 标准 Transformer：P² = 256x（L=96, P=16）

---

## 7. 实验表现

### 7.1 主要结果（ETTh1）

| 模型 | 96步 MSE | 192步 MSE | 336步 MSE | 720步 MSE |
|------|---------|----------|----------|----------|
| Transformer | 0.512 | 0.548 | 0.627 | 0.714 |
| Autoformer | 0.449 | 0.487 | 0.553 | 0.652 |
| FEDformer | 0.437 | 0.473 | 0.538 | 0.642 |
| DLinear | 0.423 | 0.458 | 0.521 | 0.634 |
| **PatchTST** | **0.413** | **0.444** | **0.501** | **0.586** |

PatchTST 在所有预测长度上超越全部 2023 年前的变体，且赢了 DLinear。预测越长优势越大（720步领先 DLinear 7.6%）。

### 7.2 消融实验

| 配置 | MSE | 显存 |
|------|-----|------|
| 标准 Transformer | 0.512 | 100% |
| + Channel-Independent | 0.468 | 33% |
| + Patching (P=16) | **0.413** | 5% |
| + 预训练 | **0.397** | 5% |

Patching 自身：MSE 降 11.7%，显存降到 1/20。

### 7.3 不同 Patch 长度

| P | Token数 | MSE | 训练时间 |
|---|---------|-----|---------|
| 1 | 96 | 0.468 | 1.0x |
| 8 | 12 | 0.421 | 0.2x |
| **16** | **6** | **0.413** | **0.1x** |
| 24 | 4 | 0.416 | 0.08x |

P=16 是精度和效率的最佳平衡。

---

## 8. 优缺点

### 优点

**1. 极其简单，效果 SOTA**：核心就 `unfold + Linear + Transformer + Linear`，200行代码。

**2. 复杂度暴降**：O(L²) → O((L/P)²)，L=336,P=16 → 降 441 倍。

**3. Channel-Independent 意外地强**：阻断噪声，支持任意变量数。

**4. 预训练在数据稀缺时救命**：少样本（<1万）提升 5-15%。

**5. 可解释性强**：每个 patch 对应具体时间段，可可视化 patch 间注意力。

### 缺点及应对

**1. 变量间关系丢失**：应对 → 后融合层，或换 Crossformer/iTransformer

**2. Patch 长度敏感**：应对 → 按数据周期手动设置

**3. Flatten Head 参数量可能大**：N_patch×d=64×512=32768 → 用均值池化压缩

**4. 预训练增加复杂度**：数据充足时可跳过

---

## 9. 模型对比

| 维度 | Transformer | FEDformer | iTransformer | **PatchTST** |
|------|-----------|-----------|-------------|-------------|
| Token | 时间步 | 时间步(频域) | 整条变量 | **连续patch** |
| Token数 | L | L | C | **L/P** |
| Attn复杂度 | O(L²) | O(L log L) | O(C²) | **O((L/P)²)** |
| 变量处理 | 混合 | 混合 | Attention | **独立** |
| 预训练 | 无 | 无 | 无 | **Masked Patch** |
| 推荐场景 | 短序列 | 强周期 | 强耦合 | **通用首选** |

---

## 10. 实践建议

### 建议1：几乎总是从 PatchTST 开始
```python
baselines = [DLinear, PatchTST]
# DLinear → 线性够不够？PatchTST → 非线性能否超过？
```

### 建议2：Patch 长度选择
```python
# 让 token 数在 4~12 之间
for p in [8, 12, 16, 24, 32]:
    if 4 <= seq_len // p <= 12:
        return p  # 默认 p=16
```

### 建议3：何时预训练
- 样本 < 5000 → 强烈推荐
- 样本 > 5万 → 可跳过

### 建议4：处理不同频率数据
```python
# 15分钟数据（日周期=96）→ P ∈ [16, 24, 32, 48]
# 日数据（周周期=7）→ P ∈ [2, 3, 4]
```

---

## 11. 关键超参数

| 参数 | 推荐范围 | 默认值 | 优先级 |
|------|---------|--------|--------|
| patch_len | 8-32 | 16 | ⭐⭐⭐⭐⭐ |
| d_model | 128-512 | 128 | ⭐⭐⭐⭐ |
| n_layers | 2-4 | 3 | ⭐⭐⭐ |
| stride | patch_len | patch_len | ⭐⭐ |
| dropout | 0.1-0.2 | 0.1 | ⭐⭐ |

---

## 12. 面试问题

**Q1: PatchTST 的核心思想？**

> Patching（时序切片→token）+ Channel-Independent（变量独立）+ 自监督预训练（Masked Patch Modeling）。

**Q2: 为什么 Patch 比逐点好？**

> (1) 单点无语义，连续段才有；(2) L→L/P，Attention 降 P² 倍；(3) 天然利用局部相关性。

**Q3: Channel-Independent 为什么反而更好？**

> 阻止跨变量噪声传播。虽丢跨变量信息，但 Patch 建模提升远超损失。变量少（C<10）时可试 Channel-Mixing。

**Q4: 标题"64 Words"什么意思？**

> L=512,P=16→32个patch。"64个词就够"强调不需要很多token。实际4~32个。

**Q5: vs iTransformer 区别？**

> PatchTST切时间维度，变量独立，注意力在patch间。iTransformer变量当token，时间交FFN。两者互补。

**Q6: 预训练有用吗？**

> 数据<1万时提升5-15%，充足时仅2-3%。遮40% patch→重建。PatchTST受益最大（patch结构天然适合）。

---

## 13. 代码资源

```bash
git clone https://github.com/yuqinie98/PatchTST.git          # 官方
git clone https://github.com/thuml/Time-Series-Library       # TSLib
```

> 配套代码：`code/04_transformer/02_patchtst_demo.py`

---

## 14. 进一步阅读

- [PatchTST 论文](https://arxiv.org/abs/2211.14730)
- [DLinear 论文](https://arxiv.org/abs/2205.13504)
- [iTransformer 论文](https://arxiv.org/abs/2310.06625)
- [BERT 论文](https://arxiv.org/abs/1810.04805)

---

## 15. 总结

PatchTST 是 2023 年时序预测最重要的论文之一。贡献不在新注意力机制，而在**重新定义"什么是 token"**。

**为什么重要？**
- 证明 patch > 逐点 token → 开创"Patch+X"范式
- 证明 Channel-Independent 没那么差 → 启发后续设计
- 把研究从"优化注意力"转向"如何表示数据"

**什么时候用？**
- ✅ 长序列预测 (L>96)
- ✅ 多变量但变量关系非关键
- ✅ 追求简单可靠的 baseline
- ⚠️ 变量关系很关键 → iTransformer 或 PatchTST+后融合

PatchTST 不是最 fancy 的模型，但是最实用的之一。

---

**下一步**：运行 `code/04_transformer/02_patchtst_demo.py`，动手实现 PatchTST！
