# TimesNet: 把 1D 时序变成 2D 图像——用卷积捕捉多周期模式

> **论文**: [TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis](https://arxiv.org/abs/2210.02186)
> **机构**: 清华大学 (ICLR 2023)
> **作者**: Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, Mingsheng Long

---

## 1. 核心思想

TimesNet 回答了一个根本问题：**时间序列中的多周期模式，能不能用计算机视觉的方法来捕捉？**

传统做法把时序当作 1D 序列处理（RNN、1D CNN、Transformer）。但真实世界的时序往往包含**多个周期**（如电力数据有日周期+周周期+年周期），这些周期在 1D 视角下互相叠加，难以分离。

TimesNet 的洞见：**把每个周期展开成 2D，让周期内的变化（列）和周期间的变化（行）分别对应 2D 图像的两个维度——然后用 2D 卷积同时捕捉两者。**

### 从一个直观类比开始

想象你在分析一整年的每日温度：

**1D 视角**（传统做法）：
```
[1月1日, 1月2日, ..., 12月31日]  →  365 个点排成一条线
很难同时看到"每天的日内变化"和"每周的周间变化"
```

**2D 视角**（TimesNet）：
```
发现日周期=24小时，把数据重排：
  Hour 0  1   2  ... 23
Day1 [ 20, 21, 23, ..., 18 ]  ← 一天内的变化（列方向）
Day2 [ 19, 21, 22, ..., 17 ]  ← 跨天变化（行方向）
Day3 [ 21, 22, 24, ..., 19 ]
...
```

现在：
- **列方向（周期内）**：每小时的变化模式 → 2D 卷积可以捕捉
- **行方向（周期间）**：每天同一时刻的变化 → 2D 卷积也可以捕捉

这就像把时序"折叠"成一张图片，然后用 CNN 分析这张图片！

### TimesNet 的三个核心步骤

**步骤1：发现周期（FFT 找 Top-K 周期）**
```
输入序列 → FFT → 频谱 → 选能量最大的 K 个频率
频率 f → 对应周期 p = L/f
例如：发现 24小时、12小时、168小时（周）等周期
```

**步驟2：1D → 2D 重塑（按每个周期折叠）**
```
1D 序列 [x1, x2, ..., x96]  周期 p=24
  → 2D tensor [4 行 × 24 列]
  Row0: [x1,  ..., x24]   ← 第1个周期的24步
  Row1: [x25, ..., x48]   ← 第2个周期的24步
  Row2: [x49, ..., x72]
  Row3: [x73, ..., x96]
```

**步骤3：2D 卷积提取模式（Inception Block）**
```
2D tensor → [2D Conv (Inception)] → 捕捉周期内+周期间模式 → 2D → 1D 展平
```

### 一句话总结

**FFT 自动发现多周期 + 按每个周期将 1D 序列重排为 2D + 2D Inception 卷积捕捉周期内/间变化 = 以视觉方法解决时序问题。**

```
Informer/Autoformer: 优化注意力复杂度 (O(L²)→O(L log L))
PatchTST:            改变 token 定义 (逐点 → patch)
TimesNet:            改变数据维度 (1D → 2D) + 用卷积替代注意力
```

---

## 2. 演进路线：从 1D 到 2D 的视角转换

```
RNN/LSTM/TCN (2015-2018) — 1D 序列建模
    │  局限: 难以捕捉长程依赖
    │
    ├─→ Transformer 系列 (2019-2023)
    │   Informer/Autoformer/FEDformer/PatchTST/iTransformer
    │   都在 1D 空间操作（1D token + 1D attention）
    │   问题: 多周期信号在 1D 视角下叠加，难以分离
    │
    ├─→ 1D CNN 方法 (如 SCINet) — 1D 卷积
    │   卷积核大小有限，感受野受限
    │
    └─→ ★ TimesNet (ICLR 2023) — 1D → 2D 重塑
        核心洞察: 周期 p 的序列 → 折叠成 (L/p) × p 的 2D 张量
        → 列 = 周期内变化（intraperiod）
        → 行 = 周期间变化（interperiod）
        → 2D 卷积同时捕捉两者！

        与 PatchTST 同时期（ICLR 2023），但思路完全不同：
        PatchTST: 1D patch → 1D token → Transformer
        TimesNet: 1D series → 2D tensor → CNN (Inception)
```

**为什么 2D 比 1D 更有优势？**

1D 视角下，相距 24 步的两个点（如今天 8:00 和明天 8:00）在序列中距离很远，1D 卷积需要很大的 kernel 才能触及。2D 视角下，它们在行方向是相邻的（上下行同一列），2D 卷积天然能捕捉。

---

## 3. 模型架构

### 3.1 整体结构

TimesNet 采用**残差堆叠**设计，由 N 个 TimesBlock 串联组成：

```
输入: X ∈ R^{B×L×C}  (L=回溯长度, C=变量数)

        ┌─────────────────────────────┐
        │  1. Embedding               │
        │  Linear: (L, C) → (L, d)    │
        └─────────────────────────────┘
                    │
        ┌───────────▼───────────┐
        │  TimesBlock × 1       │
        │  (FFT找周期→1D→2D→    │
        │   2D Conv→2D→1D)      │
        └───────────┬───────────┘
                    │  残差连接
        ┌───────────▼───────────┐
        │  TimesBlock × 2       │
        └───────────┬───────────┘
                    │
                  ...
                    │
        ┌───────────▼───────────┐
        │  TimesBlock × N       │
        └───────────┬───────────┘
                    │
        ┌───────────▼───────────┐
        │  Prediction Head      │
        │  Linear: d → H        │
        └───────────┬───────────┘
                    ▼
输出: Ŷ ∈ R^{B×H×C}
```

### 3.2 TimesBlock：核心组件

每个 TimesBlock 由两步组成：(1) FFT 找周期，(2) 按周期做 2D 卷积。

```
输入: X ∈ R^{B×L×d}

  ┌─────────────────────────────────────────────┐
  │ 步骤1: FFT 发现 Top-K 周期                     │
  │ ───────────────────────────────────────────  │
  │ 对每个通道做 FFT → 频谱 A(f)                   │
  │ 选择能量最大的 K 个频率 → 对应 K 个周期        │
  │ {p1, p2, ..., pK}                            │
  └─────────────────────────────────────────────┘
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
  周期 p1 的 2D 处理      周期 p2 的 2D 处理    ...    周期 pK
         │                     │
         ▼                     ▼
  ┌──────────────┐      ┌──────────────┐
  │ 1D → 2D 重塑  │      │ 1D → 2D 重塑  │
  │ (L,)→(p1,f1) │      │ (L,)→(p2,f2) │
  └──────────────┘      └──────────────┘
         │                     │
         ▼                     ▼
  ┌──────────────┐      ┌──────────────┐
  │ Inception    │      │ Inception    │
  │ 2D Conv      │      │ 2D Conv      │
  │ (参数共享)    │      │ (参数共享)    │
  └──────────────┘      └──────────────┘
         │                     │
         ▼                     ▼
  ┌──────────────┐      ┌──────────────┐
  │ 2D → 1D 展平  │      │ 2D → 1D 展平  │
  └──────────────┘      └──────────────┘
         │                     │
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │ 加权融合 K 个周期的输出  │
         │ (用频谱幅度做权重)      │
         └──────────┬──────────┘
                    ▼
输出: X_out ∈ R^{B×L×d}
```

### 3.3 步骤1：FFT 发现周期

这是 TimesNet 的关键第一步——自动发现数据中有哪些周期，而不是人工设定。

```python
def find_periods(x, k=5):
    """
    x: (B, L, d) — 时序特征
    返回: 能量最大的 k 个周期长度
    """
    B, L, d = x.shape
    
    # 1. FFT: (B, L, d) → (B, L//2+1, d)
    x_fft = torch.fft.rfft(x, dim=1)
    
    # 2. 振幅（能量）: 取绝对值
    amplitude = torch.abs(x_fft)  # (B, L//2+1, d)
    
    # 3. 对所有通道求平均振幅
    avg_amplitude = amplitude.mean(dim=(0, -1))  # (L//2+1,)
    
    # 4. 选择能量最大的 k 个频率（排除频率 0 = 直流分量）
    topk_amplitudes, topk_indices = torch.topk(avg_amplitude[1:], k=k)
    
    # 5. 频率 → 周期: period = L / frequency
    # 频率索引 f 对应周期 L/f
    periods = L // (topk_indices + 1)  # +1 因为排除了频率0
    
    return periods  # [p1, p2, ..., pK]
```

**一个具体例子**：
```python
L = 96  # 回溯96步（比如96小时=4天）
FFT 后发现：
  频率 f=4  → 周期=96/4=24 小时（日周期）← 能量最高!
  频率 f=8  → 周期=96/8=12 小时（半日周期）
  频率 f=2  → 周期=96/2=48 小时（双日周期）
  频率 f=1  → 周期=96/1=96 小时（4天周期）
  
TimesNet 选 top-K（K=5），自动捕捉这些周期
```

**为什么自动发现周期很重要？**
- 不依赖人工的领域知识（"用电数据日周期=24"）
- 可以适应不同采样频率（小时、分钟、天）
- 可以发现意想不到的周期（如 21 天的"伪周期"）

### 3.4 步骤2：1D → 2D 重塑

将一个周期 p 的 1D 序列折叠成 2D 张量：

```python
def reshape_1d_to_2d(x, period):
    """
    x: (B, L, d)
    period: 周期长度 p
    
    返回: (B, d, rows, cols)
    其中 rows = L/p, cols = p
    """
    B, L, d = x.shape
    
    # 计算行数和列数
    n_rows = L // period  # 有多少个完整周期
    n_cols = period       # 每个周期有多少步
    
    # 截断到周期整数倍
    length = n_rows * n_cols
    x = x[:, :length, :]
    
    # 重塑: (B, L, d) → (B, d, n_rows, n_cols)
    x = x.reshape(B, n_rows, n_cols, d)
    x = x.permute(0, 3, 1, 2)  # (B, d, n_rows, n_cols)
    
    return x
```

**可视化**：
```
1D 序列 (L=96, 周期 p=24):
  [x1, x2, x3, ..., x24, x25, ..., x48, ..., x72, ..., x96]

折叠成 2D (4行 × 24列):
        Col0    Col1    Col2    ...   Col23
  Row0  [x1,     x2,     x3,    ...,  x24]    ← 第1个24小时周期
  Row1  [x25,    x26,    x27,   ...,  x48]    ← 第2个24小时周期
  Row2  [x49,    x50,    x51,   ...,  x72]    ← 第3个24小时周期
  Row3  [x73,    x74,    x75,   ...,  x96]    ← 第4个24小时周期

列方向 (↓): 周期内变化 — 同一周期内，步与步之间的关系
            例如 Row0[Col0→Col23] 展示了24小时内温度如何变化

行方向 (→): 周期间变化 — 不同周期同一位置之间的关系
            例如 Row0[Col0]→Row1[Col0]→... 展示了每天0点的温度如何变化
```

**注意：0填充**
如果 L 不能被 p 整除，需要在末尾补 0：
```python
length = ((L - 1) // period + 1) * period  # 向上取整
x_padded = F.pad(x, (0, 0, 0, length - L))
```

### 3.5 步骤3：2D Inception 卷积

TimesNet 使用 Inception 模块（借鉴 GoogLeNet）来捕捉 2D 模式。"Inception"意味着用**多个不同大小的卷积核同时捕捉模式**：

```python
class InceptionBlock2D(nn.Module):
    """多尺度 2D 卷积 —— 同时用不同 kernel 大小捕捉模式"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 3种不同 kernel 的卷积分支（就像同时用不同分辨率的镜头看）
        self.conv1 = nn.Conv2d(in_channels, out_channels//3, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels//3, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels//3, kernel_size=5, padding=2)
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
    
    def forward(self, x):
        """x: (B, d, rows, cols)"""
        c1 = self.conv1(x)
        c3 = self.conv3(x)
        c5 = self.conv5(x)
        
        # 拼接多个分支的结果
        out = torch.cat([c1, c3, c5], dim=1)
        return self.act(self.bn(out))
```

**为什么用 Inception？**

不同周期长度导致 2D tensor 的尺寸不同：
- 周期=24 → 2D tensor = 4×24（宽而扁）
- 周期=12 → 2D tensor = 8×12（较方正）

Inception 的多尺度 kernel 可以适应不同形状的 2D tensor：
- 1×1 kernel：逐点变换
- 3×3 kernel：捕捉局部邻域模式
- 5×5 kernel：捕捉更大范围的模式

### 3.6 步骤4：2D → 1D 展平 + 多周期融合

```python
def reshape_2d_to_1d(x, original_length):
    """
    x: (B, d, rows, cols)
    返回: (B, L, d)
    """
    B, d, rows, cols = x.shape
    x = x.permute(0, 2, 3, 1)       # (B, rows, cols, d)
    x = x.reshape(B, rows * cols, d) # (B, rows*cols, d)
    
    # 截断或填充到原始长度
    if x.shape[1] > original_length:
        x = x[:, :original_length, :]
    elif x.shape[1] < original_length:
        x = F.pad(x, (0, 0, 0, original_length - x.shape[1]))
    
    return x
```

**多周期融合**：每个周期处理完后，用频谱振幅加权融合：

```python
# K 个周期各自处理，输出 K 个 1D 结果
outputs = [process_period(x, p) for p in periods]  # [out1, out2, ..., outK]

# 用 FFT 振幅做加权融合（重要周期贡献大）
amplitudes_normalized = F.softmax(amplitudes[topk_indices], dim=0)
fused_output = sum(w * out for w, out in zip(amplitudes_normalized, outputs))
```

### 3.7 完整 TimesBlock

```python
class TimesBlock(nn.Module):
    def __init__(self, seq_len, d_model, top_k=5, d_ff=32):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.top_k = top_k
        
        # Inception 2D Conv（所有周期共享参数）
        self.conv = nn.Sequential(
            InceptionBlock2D(d_model, d_ff),
            InceptionBlock2D(d_ff, d_model),
        )
        
        # 残差后接的 LayerNorm
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """
        x: (B, L, d)
        返回: (B, L, d)
        """
        B, L, d = x.shape
        residual = x
        
        # 1. FFT 找 Top-K 周期
        x_fft = torch.fft.rfft(x, dim=1)
        amplitude = torch.abs(x_fft).mean(dim=(0, -1))  # 平均所有通道
        _, topk_indices = torch.topk(amplitude[1:], k=self.top_k)
        periods = L // (topk_indices + 1)  # 频率→周期
        amplitudes_selected = amplitude[1:][topk_indices]
        
        # 2. 对每个周期做 2D 处理
        outputs = []
        for p in periods:
            p = p.item()
            if p < 2 or p > L // 2:
                continue
            
            # 填充到周期整数倍
            length = ((L - 1) // p + 1) * p
            x_padded = F.pad(x, (0, 0, 0, length - L))
            
            # 1D → 2D: (B, length, d) → (B, d, length//p, p)
            x_2d = x_padded.reshape(B, length // p, p, d)
            x_2d = x_2d.permute(0, 3, 1, 2)
            
            # 2D Conv
            x_2d = self.conv(x_2d)
            
            # 2D → 1D: (B, d, length//p, p) → (B, L, d)
            x_1d = x_2d.permute(0, 2, 3, 1)
            x_1d = x_1d.reshape(B, length, d)
            x_1d = x_1d[:, :L, :]
            
            outputs.append(x_1d)
        
        # 3. 加权融合
        if len(outputs) > 0:
            weights = F.softmax(amplitudes_selected[:len(outputs)], dim=0)
            out = sum(w * o for w, o in zip(weights, outputs))
        else:
            out = torch.zeros_like(x)
        
        # 4. 残差 + LayerNorm
        return self.norm(residual + out)
```

### 3.8 完整模型

```python
class TimesNet(nn.Module):
    def __init__(self, enc_in=7, seq_len=96, pred_len=96,
                 d_model=32, d_ff=32, top_k=5, num_blocks=2):
        super().__init__()
        
        # Embedding
        self.embedding = nn.Linear(enc_in, d_model)
        
        # 堆叠多个 TimesBlock
        self.blocks = nn.ModuleList([
            TimesBlock(seq_len, d_model, top_k, d_ff)
            for _ in range(num_blocks)
        ])
        
        # 预测头
        self.predictor = nn.Linear(seq_len * d_model, pred_len * enc_in)
        self.enc_in = enc_in
        self.pred_len = pred_len
    
    def forward(self, x):
        """
        x: (B, L, C)
        返回: (B, H, C)
        """
        B, L, C = x.shape
        
        # Embedding: (B, L, C) → (B, L, d)
        x = self.embedding(x)
        
        # TimesBlocks
        for block in self.blocks:
            x = block(x)
        
        # Flatten + 预测: (B, L, d) → (B, L*d) → (B, H*C)
        x = x.reshape(B, -1)
        x = self.predictor(x)
        
        # (B, H*C) → (B, H, C)
        x = x.reshape(B, self.pred_len, self.enc_in)
        
        return x
```

---

## 4. 关键设计决策

### 4.1 为什么是 2D 而不是 1D？

实验对比（ETTh1, L=96, H=96）：

| 方法 | MSE | 说明 |
|------|-----|------|
| 1D Conv (kernel=3~25) | 0.421 | 卷积核有限，长程依赖弱 |
| 1D Conv (kernel=51) | 0.416 | 核太大导致过拟合 |
| **2D Conv (Inception)** | **0.381** | 2D同时捕捉周期内/间 |

**直觉**：1D 卷积沿着时间轴滑动——相邻时间步很近，但跨周期的对应位置（今天8:00 vs 明天8:00）在 1D 空间中距离 p=24 步，需要很大的 kernel。2D 视角下它们在上下行的同一列，kernel=3×3 就能覆盖。

### 4.2 FFT 找周期 vs 人工设定周期

| 方法 | 优点 | 缺点 |
|------|------|------|
| 人工设定 | 利用领域知识 | 不通用，可能遗漏周期 |
| **FFT 自动** | 自适应、通用 | 可能找到伪周期（噪声） |

**TimesNet 用 top-K 限制**（默认 K=5）：只选 K 个最重要的周期，避免噪声周期。

### 4.3 2D Conv 参数共享：所有周期用同一套 Inception

为什么？不同周期的 2D tensor 形状不同（3×24, 6×12 等），但都代表"周期内+周期间变化"的同一个概念。参数共享减少了参数量，也让模型学习到"通用的周期模式"。不同周期只需 reshape 到对应形状，其余操作完全相同。

### 4.4 TimesBlock 堆叠：逐层发现更深层的周期

```
第1个 TimesBlock: 发现日周期（24h）→ 捕捉日周期模式
第2个 TimesBlock: 在残差中发现周周期（168h）→ 捕捉周周期模式
```

随着层数加深：
- 第1层：可能发现日周期、半天周期
- 第2层：可能发现周周期、月周期
- 更深层：发现更长或更精细的周期

每层的 FFT 是独立计算的——上层的残差可能包含上一层未能捕捉的模式。

---

## 5. 训练与推理

### 5.1 训练

- **非自回归**：一次前向输出 H 步
- **Loss**：MSE（论文默认）
- **学习率**：1e-3 ~ 1e-4，Adam 优化器
- **Batch Size**：16~64

### 5.2 推理

- 单次前向 = 全部预测
- 主要开销：K 次 FFT + K 次 2D Conv
- Top-K 不大（K=5），2D 尺寸也小，推理成本可控

### 5.3 复杂度分析

对于单个 TimesBlock：
```
FFT: O(L log L ⋅ d)
2D Conv (K 个周期): O(K ⋅ d ⋅ rows ⋅ cols) = O(K ⋅ d ⋅ L)
总复杂度: O(L log L ⋅ d + K ⋅ d ⋅ L) ≈ O(K ⋅ d ⋅ L)
```

对比标准 Transformer 的 O(L²·d)：TimesNet 在长序列（L>100）时有明显优势。

---

## 6. 实验表现

基于论文报告（ETTh1, 预测长度 96~720）：

| 模型 | 96步 MSE | 192步 MSE | 336步 MSE | 720步 MSE |
|------|---------|----------|----------|----------|
| Transformer | 0.512 | 0.548 | 0.627 | 0.714 |
| FEDformer | 0.437 | 0.473 | 0.538 | 0.642 |
| DLinear | 0.423 | 0.458 | 0.521 | 0.634 |
| PatchTST | 0.413 | 0.444 | 0.501 | 0.586 |
| **TimesNet** | **0.381** | **0.409** | **0.465** | **0.554** |

**关键观察**：TimesNet 在所有预测长度上超越了同时代（2023年初）的所有方法，包括 FEDformer 和 DLinear。它甚至在某些设置下超越 PatchTST。

### 消融实验

| 配置 | ETTh1 MSE |
|------|----------|
| TimesNet 完整 | **0.381** |
| 只用1个周期（top-K=1）| 0.403 |
| 不用 Inception（单 kernel=3）| 0.398 |
| 1D Conv 替代 2D Conv | 0.421 |
| 不用 FFT（固定周期=24）| 0.392 |

**结论**：FFT+多周期+Inception 2D 三者缺一不可，但 FFT 自动发现的贡献最大。

### 多任务能力

TimesNet 不仅擅长预测，论文展示了5项任务：
- ✅ 长期预测（forecasting）
- ✅ 短期预测
- ✅ 填补缺失值（imputation）
- ✅ 异常检测（anomaly detection）
- ✅ 分类（classification）

这种多任务通用性是 TimesNet 的一大亮点，因为它捕捉的是"通用的周期模式"，不局限于某一类任务。

---

## 7. 优缺点

### 优点

**1. 多周期自动发现**

FFT 自动找周期，不需要领域知识，适用于任何采样频率。

**2. 1D→2D 视角转换是根本性创新**

不是优化现有方法（如优化 Attention），而是改变数据的组织形式。2D 视角下，周期内和周期间的变化被解耦，各自更容易被 CNN 捕捉。

**3. 多任务通用**

同一架构在预测、填补、异常检测、分类上都有效，证明了多周期建模的通用性。

**4. 计算高效**

没有显式的 Attention，2D Conv 非常快。O(K·d·L) vs Transformer 的 O(L²·d)。

**5. Inception 适应不同周期**

不同周期 → 不同 2D 尺寸 → Inception 多尺度 kernel 天然适应。

### 缺点及应对

**1. FFT 的边界效应**

序列长度不是周期整数倍 → 频谱泄漏。应对：用 Hann 窗函数减轻。

**2. 对非周期信号不敏感**

白噪声、随机游走 → FFT 找不到周期 → 退化为普通 2D Conv。应对：非周期场景考虑 DLinear/iTransformer。

**3. 填充策略影响性能**

L 不能整除周期 p → 需要 padding。Padding 0 会引入虚假信号。应对：尽可能选 L 为常见周期的公倍数。

**4. top-K 是敏感超参**

K 太小漏周期，K 太大引入噪声。默认 K=5 是经验值。

**5. 变量间交互弱**

和 PatchTST 类似，不同变量的时序独立处理。应对：需要变量交互时用 Crossformer 或 iTransformer。

---

## 8. 与其他模型的对比

| 维度 | FEDformer | PatchTST | TimesNet |
|------|-----------|----------|----------|
| 核心思路 | 频域增强注意力 | Patch→token→Transformer | **1D→2D+CNN** |
| 周期处理 | FFT随机选M个频率 | Patch隐含局部周期 | **FFT选K个周期+2D重塑** |
| 时序建模 | 频域乘法 | Self-Attention(patch间) | **2D Inception Conv** |
| 多任务 | 仅预测 | 预测+预训练 | **预测+填补+异常+分类** |
| 可解释性 | 频谱图(好) | 注意力图(好) | **2D激活图+频谱(最好)** |
| 复杂度 | O(L log L) | O((L/P)²) | **O(K·d·L)** |

**选择指南**：
- **FEDformer**：强周期数据，需要频域分析
- **PatchTST**：通用首选，简单可靠
- **TimesNet**：多周期数据，需要多任务能力，追求最强周期建模

---

## 9. 实践建议

### 建议1：选择合适的 L（回溯长度）

TimesNet 对 L 有额外要求——需要能被多个周期整除。

```python
# 推荐：L 选择常见周期的公倍数
# 小时数据: L ∈ [96, 168, 336, 720]  (24的倍数)
# 日数据:   L ∈ [28, 56, 84, 168]    (7的倍数)
# 分钟数据: L ∈ [1440, 2880]         (1440=24*60)
```

### 建议2：FFT 结果的"视觉确认"

```python
# 可视化频谱，确认 FFT 找到的周期是否合理
import matplotlib.pyplot as plt

x_fft = torch.fft.rfft(data, dim=1)
amplitude = torch.abs(x_fft).mean(dim=(0, -1))

plt.stem(amplitude[1:50].numpy())  # 前50个频率
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.title('Spectrum — 峰值对应周期')
# 观察：振幅高的频率 = 重要周期
```

### 建议3：top-K 的调参

```python
# K=3: 保守，只取最确定的周期
# K=5: 默认，论文推荐
# K=10: 激进，适合复杂多周期数据

# 实验测试
for K in [3, 5, 7, 10]:
    model = TimesNet(top_k=K)
    val_loss = train_and_eval(model)
    print(f"K={K}: val_loss={val_loss:.4f}")
```

### 建议4：多任务场景优先考虑

如果你需要同时做预测+异常检测+填补，TimesNet 的通用架构有天然优势——同一个预训练模型可以用于多个下游任务。

### 建议5：非周期数据的 fallback

```python
# 检测是否适合 TimesNet
x_fft = torch.fft.rfft(data, dim=1)
amplitude = torch.abs(x_fft).mean(dim=(0, -1))

# 如果频谱很平坦（无明显峰值），说明缺乏周期性
peak_ratio = amplitude.max() / amplitude.mean()
if peak_ratio < 3:
    print("警告: 频谱无明显峰值，建议用 PatchTST 或 DLinear")
```

---

## 10. 关键超参数

| 参数 | 推荐范围 | 默认值 | 优先级 |
|------|---------|--------|--------|
| top_k | 3-10 | 5 | ⭐⭐⭐⭐⭐ |
| d_model | 16-128 | 32 | ⭐⭐⭐ |
| d_ff | 16-64 | 32 | ⭐⭐⭐ |
| num_blocks | 1-3 | 2 | ⭐⭐⭐ |
| seq_len | 96-720 | 96 | ⭐⭐⭐⭐ |

**注意**：TimesNet 的 d_model 通常比 Transformer 小（32-64 vs 128-512），因为 2D Conv 的方式更高效。

---

## 11. 面试问题

**Q1: TimesNet 的核心思想？**

> 把 1D 时间序列按周期折叠成 2D 张量，用 2D 卷积同时捕捉周期内变化（列方向）和周期间变化（行方向）。FFT 自动发现 Top-K 周期，Inception 多尺度卷积捕捉模式，加权融合多周期输出。

**Q2: 为什么 1D→2D 能提升性能？**

> 1D 视角下跨周期对应点（今天8:00→明天8:00）距离 p 步，需要大 kernel。2D 视角下它们在上下行同一列，小 kernel（3×3）就能覆盖。同时列方向捕捉短期波动，行方向捕捉长期趋势——两个维度各有明确语义。

**Q3: FFT 在 TimesNet 中的作用？**

> 自动发现数据中的主要周期，无需人工设定。不同采样频率和数据类型的周期不同，FFT 让模型自适应。

**Q4: 不同周期用同一个 Inception 合理吗？**

> 合理。所有周期的 2D tensor 都代表"周期内+周期间变化"的同一个概念，只是尺寸不同。Inception 的多尺度 kernel（1×1, 3×3, 5×5）能适应不同尺寸。参数共享减少了参数量。

**Q5: TimesNet vs PatchTST 的区别？**

> PatchTST：1D patch→Transformer Attention，变量独立。TimesNet：1D→FFT找周期→2D重塑→2D Conv，变量在 d_model 维度混合。核心差异是"用什么建模"（Attention vs Conv）和"在什么空间"（1D vs 2D）。

**Q6: TimesNet 的多任务能力从哪来？**

> 多周期模式是时序数据的通用底层结构，不依赖于特定任务（预测/填补/异常检测）。TimesNet 学习的是这种通用周期表示，因此可以迁移到不同任务。

---

## 12. 代码资源

```bash
git clone https://github.com/thuml/TimesNet.git            # 官方
git clone https://github.com/thuml/Time-Series-Library     # TSLib
```

> 配套代码：`code/02_convolution/02_timesnet_demo.py`

---

## 13. 进一步阅读

- [TimesNet 论文](https://arxiv.org/abs/2210.02186)
- [官方代码](https://github.com/thuml/TimesNet)
- [PatchTST 论文](https://arxiv.org/abs/2211.14730) — ICLR 2023 同期，思路迥异
- [FEDformer 论文](https://arxiv.org/abs/2201.12740) — FFT 在时序中的早期应用
- [Inception (GoogLeNet) 论文](https://arxiv.org/abs/1409.4842) — Inception 模块的起源

---

## 14. 总结

TimesNet 是 2023 年最具原创性的时序模型之一。它的贡献不在于优化注意力，而在于**改变数据的表示维度**——从 1D 序列变为 2D 张量，让卷积神经网络在时序数据上发挥了意想不到的效果。

**为什么重要？**
- 1D→2D 视角转换是根本性创新（不是改进注意力，而是重塑数据）
- 证明了"多周期"是时序数据的通用底层结构
- 多任务通用：一个架构解决预测、填补、异常检测、分类

**什么时候用？**
- ✅ 数据有多个明显周期（电力、天气、交通）
- ✅ 需要多任务能力（同时预测+异常检测）
- ✅ 追求可解释性（频谱+2D激活图）
- ❌ 弱周期或非周期数据（金融高频）
- ❌ 对 d_model 要求很大（TimesNet 的 d_model 通常在 32-64）

TimesNet 打开了"用视觉方法解决时序问题"的大门，是对时序分析视角的根本性拓展。

---

**下一步**：运行 `code/02_convolution/02_timesnet_demo.py`，动手实现 TimesNet！


