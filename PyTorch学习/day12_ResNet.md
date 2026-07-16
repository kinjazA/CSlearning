# 01 ResNet 背景

## 01.1 核心问题：模型退化

在 ResNet 之前，大家普遍认为网络越深效果越好。VGG 把网络堆到 19 层，GoogLeNet 堆到 22 层，但很快发现一个奇怪的现象：

```css
实验结果（CIFAR-10）：

20 层普通网络：训练误差 ≈ 0.75%，测试误差 ≈ 8.75%
56 层普通网络：训练误差 ≈ 1.92%，测试误差 ≈ 10.96%

结论：56 层的训练误差比 20 层还大！
```

这不是过拟合（过拟合是训练误差低、测试误差高），而是连训练集上的效果都变差了，是模型退化问题。说明深层网络根本没有被有效训练



反向传播时，梯度从最后一层往前传，每经过一层都会衰减：

```css
假设每层梯度平均衰减到 70%：

第20层收到的梯度：1.000
第15层收到的梯度：1.000 × 0.7^5  ≈ 0.168
第10层收到的 ≈ 0.005
第1层 收到的梯度：1.000 × 0.7^20 ≈ 0.001

第1层的参数几乎收不到任何梯度信号，无法学习。
网络越深，浅层参数越"僵死"。
```



但是，实际上，到2015年ResNet提出时，梯度消失问题已经基本被以下两件事解决了：

- **更好的初始化**（Xavier 初始化，Kaiming 初始化）
- **Batch Normalization**（2015年3月提出，比ResNet早几个月）

恺明大佬的团队也意识到，加了BN之后，梯度消失/爆炸已经不再是主要障碍，但退化问题依然存在



**核心问题：** 为什么深层网络的效果还不如浅层网络？

按理说，一个 56 层的网络，哪怕后 36 层什么都不做（直接把输入传给输出），也就是恒等映射，效果也应该至少和 20 层一样好。

但普通网络做不到"什么都不做"，原因如下：

```css
如果你希望某一层的输出等于输入（什么都不改变）：

普通网络的做法：
  output = conv2(conv1(x))
  要让两层卷积的组合 = 恒等变换
  这需要把权重精确调整到一个非常特殊的值
  从随机初始化出发，梯度要做大量工作才能找到这个值
  网络层数越多，这种调整越困难
```

说明：深层网络很难通过单纯的堆叠卷积层来学习“恒等映射”



恺明大佬的想法：**换一种学习目标，让"什么都不做"变成最容易的事。**

```css
残差网络的做法：
  output = F(x) + x
  要让输出等于输入，只需要 F(x) = 0
  让卷积输出全零
  只需要把权重初始化为接近零的小值——PyTorch 的默认初始化本来就是小随机数！
  几乎不需要训练就能做到
```

==我觉着这里也隐含着另外一个思想，就是说相邻层之间的特征图应该是比较像的，也就是说在非常深的网络中，相邻层之间的特征其实是非常相似的。所以相邻层之间不应该大变动，而应该是逐步“修正”调整这样的一个思路。==

如果把 x看作是当前的**知识状态**，那么：

- **普通网络**：y=F(x)。每一层都试图重新定义知识。如果这一层没学好，之前的知识 x就会丢失
- **ResNet**：y=x+F(x)。知识 x被直接传到了下一层，而 F(x)只是对现有知识的一个**补充或修正**

**这就像是：**

- **普通网络**：每读一个年级，就把上一年级的书全烧了，重新学一套新理论
- **ResNet**：每升一个年级，都保留着上一年级的教材，新学期只学习补充章节



**让堆叠的非线性层直接拟合恒等映射 H(x)=x在优化上非常困难**，但如果显式地把恒等映射拆出来，让网络只学习残差 F(x)=H(x)−x，当最优解接近恒等映射时，网络只需让 F(x)→0，让权重趋近于零远比让其编码一个复杂的恒等变换容易得多。

==所以，ResNet 的出发点是**让深层网络的优化更容易**，而不是解决梯度消失。梯度流的改善是一个**副产品**。==

## 01.2 对梯度消失的改善

### 1. 普通神经网络的反向传播

设网络共 $L$ 层，第 $l$ 层的输出为 $x_{l+1} = \sigma(W_l x_l)$（$\sigma$ 是激活函数），损失函数对第 $l$ 层输入的梯度由链式法则给出：

$$
\frac{\partial \mathcal{L}}{\partial x_l}
=
\frac{\partial \mathcal{L}}{\partial x_L}
\cdot
\prod_{k=l}^{L-1}
\frac{\partial x_{k+1}}{\partial x_k}
$$

每一项

$$
\frac{\partial x_{k+1}}{\partial x_k}
=
W_k \cdot \mathrm{diag}\bigl(\sigma'(z_k)\bigr)
$$

是一个矩阵。当网络很深时，这是一个连乘积。如果这些矩阵的谱范数普遍小于 $1$，连乘之后趋向于 $0$，梯度消失；如果大于 $1$，趋向于无穷，梯度爆炸。

对于 ReLU 激活，$\sigma'(z)$ 只有 $0$ 或 $1$ 两个值（死神经元会让部分路径的梯度直接变 $0$），连乘之后结构性地退化。

### 2. 残差神经网络的反向传播

对于一个残差块：$x_{l+1} = x_l + F(x_l, W_l)$，对 $x_l$ 求偏导：

$$
\frac{\partial x_{l+1}}{\partial x_l}
=
\mathbf{I} + \frac{\partial F}{\partial x_l}
$$

注意这里多出了一个单位矩阵 $\mathbf{I}$。现在把多个残差块串起来，考虑从第 $l$ 层到第 $L$ 层的雅可比：

$$
\frac{\partial x_L}{\partial x_l}
=
\prod_{k=l}^{L-1} (\mathbf{I} + J_k)
$$

其中

$$
J_k = \frac{\partial F(x_k)}{\partial x_k}
$$

把这个连乘展开（类似二项式展开）：

$$
\frac{\partial x_L}{\partial x_l}
=
\mathbf{I}
+ \sum_k J_k
+ \sum_{k<j} J_k J_j
+ \cdots
$$

这个展开式里，第一项永远是单位矩阵 $\mathbf{I}$，它代表梯度可以不经过任何权重矩阵，直接从第 $L$ 层传回第 $l$ 层。即使所有 $J_k$ 都趋近于 $0$，这一项也能保证梯度不消失。

因此，损失对第 $l$ 层的梯度为：

$$
\frac{\partial \mathcal{L}}{\partial x_l}
=
\frac{\partial \mathcal{L}}{\partial x_L}
\cdot
\left( \mathbf{I} + \text{来自各路径的贡献} \right)
$$

括号里的 $\mathbf{I}$ 是一条高速公路，梯度无论如何都能直接从输出层流回任意中间层，不需要经过任何可学习参数的连乘。

### 3. 路径集成视角

2016年 Veit 等人的论文从另一个角度解释了这件事。对于 n个残差块，在前向传播时，每个块可以走"shortcut 路径"（加法中的 x）或"残差路径"（加法中的 F(x)），所有组合构成 2^n 条从输入到输出的路径，相当于 2^n 个不同深度的子网络的**隐式集成**。

在反向传播时，梯度从2^n条路径反向流动。其中最短的路径（全走 shortcut，一步从输出回到输入）梯度衰减最少，能给早期层提供强烈的梯度信号。即使深层路径的梯度消失了，浅层路径仍然在传递有效信号。这与 Dropout + 集成的思想有异曲同工之处，但 ResNet 是在结构上天然形成的

# 02 核心组件

这里以ResNet的各个模块顺序来记的笔记。

ResNet 把整个网络（除了 Stem 和分类头）分成了四段，每段叫一个 Stage。每个 Stage 内部的所有 Block 处理的特征图尺寸和通道数完全一致，可以无缝串联。



以 ResNet-50 为例，数据从 Stem 出来是 `[B, 64, 56, 56]`，然后依次流过：

```css
Stage 1（layer1）：3 个 Bottleneck，planes=64，stride=1
  输入  [B,   64, 56, 56]
  输出  [B,  256, 56, 56]   ← 通道数扩张（64×4=256），空间尺寸不变

Stage 2（layer2）：4 个 Bottleneck，planes=128，stride=2
  输入  [B,  256, 56, 56]
  输出  [B,  512, 28, 28]   ← 通道数翻倍，空间尺寸减半

Stage 3（layer3）：6 个 Bottleneck，planes=256，stride=2
  输入  [B,  512, 28, 28]
  输出  [B, 1024, 14, 14]   ← 通道数翻倍，空间尺寸减半

Stage 4（layer4）：3 个 Bottleneck，planes=512，stride=2
  输入  [B, 1024, 14, 14]
  输出  [B, 2048,  7,  7]   ← 通道数翻倍，空间尺寸减半

```

**每个 Stage 结束，通道数翻倍，空间尺寸减半**（Stage 1 除外，它只扩通道不缩尺寸，因为 Stem 已经做过两次下采样了）。

## 02.1 Stem 入口模块

Stem 是整个网络的第一个模块，不包含任何残差连接，由一个大卷积和一个最大池化串联构成：

```css
输入：[B, 3, 224, 224]
  ↓  Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
[B, 64, 112, 112]
  ↓  BatchNorm2d + ReLU
  ↓  MaxPool2d(kernel_size=3, stride=2, padding=1)
[B, 64, 56, 56]
  → 进入 Stage 1
```

两次 stride=2 的下采样（conv 一次、maxpool 一次），将空间尺寸从 $224 \times 224$ 迅速压缩到 $56 \times 56$，面积缩小为原来的 $\frac{1}{16}$

> * 7×7 的卷积核在第一层能够捕捉较大范围内的低级特征（边缘、色彩梯度、纹理），而且此时输入通道数只有 3，计算量尚可接受。使用大卷积核的传统来自 AlexNet 的第一层（11×11）和 ZFNet（7×7），ResNet 沿用了这个做法。后续的研究（如 ResNet-D、EfficientNet）发现把这个 7×7 换成三个连续的 3×3（stride 分别为 2、1、1）参数量相近但表达能力更强

```python
import torch
import torch.nn as nn


class Stem(nn.Module):
    """
    ResNet 入口模块（Stem）。
    
    负责将高分辨率输入图像快速下采样到适合后续 Stage 处理的尺寸，
    同时将通道数从 3（RGB）扩展到 64。
    
    输入形状：[B, 3,  224, 224]
    输出形状：[B, 64,  56,  56]
    """

    def __init__(self):
        super(Stem, self).__init__()

        # 7×7 大卷积：stride=2 完成第一次空间下采样（224 → 112）
        # bias=False：后接 BN，BN 的 beta 参数功能等同于 bias，加了是冗余参数
        # padding=3：保证输出尺寸精确减半，不丢边缘信息
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)  # inplace=True：原地操作，节省显存

        # MaxPool：stride=2 完成第二次空间下采样（112 → 56）
        # padding=1：保证输出尺寸精确减半，避免边缘像素被丢弃
        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )

    def forward(self, x):
        x = self.conv1(x)    # [B, 3,  224, 224] → [B, 64, 112, 112]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # [B, 64, 112, 112] → [B, 64,  56,  56]
        return x
```

## 02.2 BasicBlock（基础残差块）

用于ResNet-18 / 34等较浅层的ResNet

BasicBlock 是 ResNet 中最基础的残差单元，由两个 $3 \times 3$ 卷积构成主路径，搭配一条 shortcut：

```css
输入 x: [B, C_in, H, W]
│
├─── shortcut 路径 ──────────────────────────────────────────┐
│    · 若 C_in == C_out 且 stride == 1：直接恒等（不做任何操作） │
│    · 否则：1×1 Conv(C_in→C_out, stride=s) + BN             │
│                                                           │
└─── 主路径：                                                │
     Conv2d(C_in, C_out, 3×3, stride=s, padding=1)          │
     → BN → ReLU                                            │
     → Conv2d(C_out, C_out, 3×3, stride=1, padding=1)       │
     → BN                                                   │
     │                              （注意：这里不接 ReLU）    │
     └──────────────────── + ←─────────────────────────────┘
                           │
                         ReLU      ← 残差加法之后统一激活
                           │
     输出: [B, C_out, H/s, W/s]
```

==这个加法就是每个位置上的数值进行相加，逐元素相加，所以要求主路径和shortcut路径的数据尺寸完全一致，也就是C_in == C_out。如果主路径里卷积调整了通道数或是尺寸，那么shortcut路径就要做调整，这里是通过1×1 Conv来实现的，调整它的参数stride就可以实现尺寸的变化，stride=1时就不变化尺寸，只改变通道数量==



==加BN是因为主路径里最后就是有个BN，所以shortcut要匹配，数值范围上要对齐==



> * 所有卷积层都不加 bias

```python
nn.Conv2d(..., bias=False)
```

紧跟在卷积之后的 BN 层会对每个通道做 $\hat{x} = \frac{x - \mu}{\sigma}$，然后用可学习的 $\gamma$ 和 $\beta$ 重新缩放和偏移。这里的 $\beta$ 参数的功能与卷积的 bias 完全等价（都是对特征图加一个可学习的常数偏移）。如果两者同时存在，卷积的 bias 会被 BN 的减均值操作完全消除，成为无效参数，徒增内存占用。

```python
class BasicBlock(nn.Module):
    """
    ResNet-18 / ResNet-34 使用的基础残差块。

    核心结构：两个 3×3 卷积构成主路径，加上一条 shortcut 连接。
    参数量（以 C 通道为例）：2 × (3×3×C×C) = 18C²

    Attributes:
        expansion (int): 输出通道相对于 planes 的倍数，BasicBlock 为 1（不扩张）。
    """

    expansion = 1  # 输出通道 = planes × expansion = planes

    def __init__(
        self,
        in_channels: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module = None
    ):
        """
        Args:
            in_channels: 输入特征图的通道数。
            planes:      本 block 的基础通道数，输出通道 = planes × expansion。
            stride:      第一个卷积的步长，stride=2 时同时完成空间下采样。
            downsample:  shortcut 的投影层（维度不匹配时由 _make_layer 传入）。
                         若为 None，表示 shortcut 直接恒等相加，无需投影。
        """
        super(BasicBlock, self).__init__()

        # ── 主路径 ────────────────────────────────────────────────────────
        # 第一个卷积：可能改变空间尺寸（stride）和通道数（in_channels → planes）
        self.conv1 = nn.Conv2d(
            in_channels, planes,
            kernel_size=3, stride=stride, padding=1,
            bias=False  # BN 的 beta 已等效于 bias，此处不需要
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        # 第二个卷积：保持空间尺寸和通道数不变（stride 固定为 1）
        self.conv2 = nn.Conv2d(
            planes, planes,
            kernel_size=3, stride=1, padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        # 注意：这里不接 ReLU，留到残差相加后统一激活

        # ── shortcut 路径 ─────────────────────────────────────────────────
        # 由外部 _make_layer 决定是否需要投影，本 block 不做判断
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x  # 保存输入，用于后续残差相加

        # ── 主路径前向 ───────────────────────────────────────────────────
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # 此处不加 ReLU，保留负值特征，等待与 shortcut 相加后统一激活

        # ── shortcut 前向 ────────────────────────────────────────────────
        if self.downsample is not None:
            identity = self.downsample(x)  # 投影：统一通道数和空间尺寸

        # ── 残差相加 + 激活 ───────────────────────────────────────────────
        out = out + identity  # 核心：残差相加，F(x) + x
        out = self.relu(out)  # 相加后统一做非线性激活

        return out
```

## 02.3 Bottleneck（瓶颈残差块）

适用于ResNet-50 / 101 / 152等层数较深的ResNet

当网络深度超过 50 层时，如果继续使用 BasicBlock，参数量和计算量会快速膨胀到不可接受的程度。以 Stage 3（256 通道）中的一个块为例做对比：

|    结构    |                         计算组成                         |          参数量           |
| :--------: | :------------------------------------------------------: | :-----------------------: |
| BasicBlock |                  2 × Conv(3×3, 256→256)                  | 2×3×3×256×256 ≈ **1.18M** |
| Bottleneck | Conv(1×1, 256→64) + Conv(3×3, 64→64) + Conv(1×1, 64→256) |        ≈ **0.07M**        |

Bottleneck 的参数量约是 BasicBlock 的 **6%**，但由于使用了更多的层数（3层 vs 2层），整体网络的表达能力反而更强。这就是 Bottleneck 的核心价值：**用更少的参数换来更深的网络深度。**



==就是"降维 → 卷积 → 升维"的三段式。为了压缩参数量，先对特征图的通道数量(256)进行降维(64)，不改变尺寸大小，然后再经过一个卷积层，然后再通过1*1卷积还原回原来的通道数量(256)==

```css
高维空间（256通道）
      ↓  1×1 Conv（降维，压缩到 64 通道）← 在高维空间只做线性投影，不做空间操作
低维瓶颈（64通道）
      ↓  3×3 Conv（空间特征提取）        ← 在低维瓶颈上做昂贵的空间卷积，大幅省参
低维空间（64通道）
      ↓  1×1 Conv（升维，还原到 256 通道）← 再投影回高维，与 shortcut 对齐维度
高维空间（256通道）
```

1×1 Conv 本质上是在通道维度上做线性组合，不涉及空间位置的交互，计算成本极低。它把昂贵的 3×3 空间卷积"包裹"在两个廉价的通道变换中间，整体上达到了既能提取空间特征又能控制参数量的效果

```python
class Bottleneck(nn.Module):
    """
    ResNet-50 / ResNet-101 / ResNet-152 使用的瓶颈残差块。

    核心结构：1×1 降维 → 3×3 空间卷积 → 1×1 升维，加上 shortcut 连接。
    参数量（以瓶颈维度 C 和扩张倍数 4 为例）：
        1×1×C×(4C) + 3×3×C×C + 1×1×C×(4C) = 17C²（远小于 BasicBlock 的 18×(4C)²）

    Attributes:
        expansion (int): 输出通道相对于 planes 的倍数，Bottleneck 为 4。
                         即输入 planes=64 时，输出通道为 256。
    """

    expansion = 4  # 输出通道 = planes × 4

    def __init__(
        self,
        in_channels: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module = None
    ):
        """
        Args:
            in_channels: 输入通道数（通常是 planes × 4，即上一个 block 的输出）。
            planes:      瓶颈通道数，1×1 降维后的通道数，3×3 卷积在这个维度上操作。
            stride:      下采样步长，放在 3×3 卷积上（而非第一个 1×1）。
            downsample:  shortcut 投影层，维度不匹配时使用。
        """
        super(Bottleneck, self).__init__()

        # ── 1×1 降维 ──────────────────────────────────────────────────────
        # 将通道从 in_channels（如 256）压缩到 planes（如 64）
        # kernel_size=1 不改变空间尺寸，纯粹做通道维度的线性组合
        self.conv1 = nn.Conv2d(in_channels, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)

        # ── 3×3 空间卷积 ──────────────────────────────────────────────────
        # 在低维瓶颈（planes 通道）上做空间特征提取，计算量大幅降低
        # stride 放在这里：让下采样发生在有空间感知的层，信息损失更少
        self.conv2 = nn.Conv2d(
            planes, planes,
            kernel_size=3, stride=stride, padding=1,
            bias=False
        )
        self.bn2   = nn.BatchNorm2d(planes)

        # ── 1×1 升维 ──────────────────────────────────────────────────────
        # 将通道从 planes 扩张到 planes × expansion（如 64 → 256），与 shortcut 对齐
        # 末尾不接 ReLU，等待残差加法后统一激活
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion,
            kernel_size=1, bias=False
        )
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)

        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # ── 主路径前向 ───────────────────────────────────────────────────
        out = self.relu(self.bn1(self.conv1(x)))    # 1×1 降维 → BN → ReLU
        out = self.relu(self.bn2(self.conv2(out)))  # 3×3 卷积 → BN → ReLU
        out = self.bn3(self.conv3(out))             # 1×1 升维 → BN（无 ReLU！）

        # ── shortcut 前向 ────────────────────────────────────────────────
        if self.downsample is not None:
            identity = self.downsample(x)

        # ── 残差相加 + 激活 ───────────────────────────────────────────────
        out = out + identity
        out = self.relu(out)

        return out
```

## 02.4 Pre-activation Block（ResNet v2）

### v1 的问题：shortcut 上的 ReLU 打断了恒等映射

在原版 ResNet（v1）中，残差相加之后还有一个 ReLU：

```
v1 的结构：
x → [Conv → BN → ReLU → Conv → BN] → (+x) → ReLU → 输出
                                                ↑
                                    这个 ReLU 作用在了 shortcut 路径上
```

从梯度流的角度看，这个最终的 ReLU 会对相加后的结果做截断，负值部分被清零。这意味着 shortcut 不再是一条纯粹的恒等路径——它的输出被 ReLU 修改了，传递回来的梯度也因此被截断。从数学上说，v1 中真正的恒等连接并不完全"干净"

### v2 的改动：BN/ReLU 前置，shortcut 纯净

2016 年，何恺明提出了改进版（v2），称为 **Pre-activation ResNet**。核心思路是把 BN 和 ReLU 移到卷积之前（因此叫 pre-activation），让残差相加的输出直接作为下一个 block 的输入，而不经过任何激活函数：

```
v2 的结构：
x → [BN → ReLU → Conv → BN → ReLU → Conv] → (+x) → 输出
                                              ↑
                         shortcut 上只有纯粹的加法，没有任何激活
```

这样，shortcut 路径上只有一个干净的 $+x$，梯度可以完全无阻地沿 shortcut 传回，理论上梯度流比 v1 更顺畅

### 适用场景：极深网络收益更明显

在常规深度（50/101 层）上，v2 与 v1 的性能差异很小（在 ImageNet 上不到 0.5%）。但在极深网络（如 1001 层 ResNet）的实验中，v2 有明显优势，训练也更稳定。在日常使用中，torchvision 提供的标准 ResNet 仍然是 v1 结构；如果你在做超深网络的实验或者科研，可以考虑切换到 v2

```python
class PreActBottleneck(nn.Module):
    """
    Pre-activation Bottleneck（ResNet v2）。

    与标准 Bottleneck (v1) 的关键区别：
      v1 顺序：Conv → BN → ReLU（激活在卷积之后）
      v2 顺序：BN → ReLU → Conv（激活在卷积之前，即 Pre-activation）

    好处：shortcut 路径上是纯粹的恒等加法，没有任何激活函数打断，
         梯度可以完全无损地从输出层直通到任意浅层。

    Attributes:
        expansion (int): 输出通道倍数，与标准 Bottleneck 相同，为 4。
    """

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module = None
    ):
        super(PreActBottleneck, self).__init__()

        # ── Pre-activation 结构：BN 和 ReLU 在 Conv 之前 ─────────────────
        self.bn1   = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, planes, kernel_size=1, bias=False)

        self.bn2   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes,
            kernel_size=3, stride=stride, padding=1, bias=False
        )

        self.bn3   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion,
            kernel_size=1, bias=False
        )

        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # ── 先激活，再卷积（Pre-activation 核心）──────────────────────────
        out = self.conv1(self.relu(self.bn1(x)))          # BN → ReLU → 1×1 降维
        out = self.conv2(self.relu(self.bn2(out)))        # BN → ReLU → 3×3 卷积
        out = self.conv3(self.relu(self.bn3(out)))        # BN → ReLU → 1×1 升维

        if self.downsample is not None:
            # 注意：v2 中 downsample 也应使用 pre-activation 风格，
            # 但为简洁起见，此处仍用 1×1 Conv + BN，实际使用时需注意。
            identity = self.downsample(x)

        # shortcut 上只有纯粹的加法，无任何激活函数
        return out + identity
```

## 02.5 Downsample 模块（shortcut 投影层）

当一个 Stage 的第一个 block 需要同时改变通道数和空间尺寸时，shortcut 的直接恒等相加就不可行了（形状不匹配无法相加）。论文提出了三种处理方案：

**方案 A：Zero-padding shortcut（零填充）**

当通道数增加时，对新增的通道用零填充补齐，不引入任何新参数。缺点是新增通道全是零，没有任何有效信息，梯度也无法通过这些通道回传，相当于这部分 shortcut 是"死的"

**方案 B：仅在维度不匹配时用 1×1 Conv 投影（论文默认方案）**

只有当 stride≠1 或通道数不匹配时才使用 1×1 Conv 投影，其他情况走恒等。这是性价比最高的方案，也是 torchvision 中 ResNet 的默认实现

**方案 C：所有 shortcut 一律用 1×1 Conv 投影**

包括维度不变的情况也强制走投影。论文实验表明方案 C 比方案 B 有微小的精度提升，但参数量更大，通常认为提升不值得这个代价

消融实验结论：方案 C > 方案 B > 方案 A，但差距很小，方案 B 是最常用的实践选择

```python
def build_downsample(
    in_channels: int,
    out_channels: int,
    stride: int
) -> nn.Module:
    """
    构建 shortcut 投影层（方案 B：仅在维度不匹配时创建）。

    触发条件：stride != 1（需要空间下采样）或 in_channels != out_channels（通道数变化）。
    两个条件任满足其一就需要投影，否则返回 None（恒等连接）。

    Args:
        in_channels:  shortcut 输入的通道数。
        out_channels: shortcut 输出的通道数（应与主路径输出一致）。
        stride:       下采样步长。

    Returns:
        nn.Sequential（含 1×1 Conv + BN）或 None。
    """
    if stride != 1 or in_channels != out_channels:
        return nn.Sequential(
            # 1×1 Conv：同时完成通道对齐和空间下采样
            # kernel_size=1 不改变感受野，只做线性投影
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=1,
                stride=stride,   # 与主路径的下采样步长一致，保证空间尺寸对齐
                bias=False       # 后接 BN，bias 冗余
            ),
            # BN：标准化 shortcut 的输出，与主路径末尾的 BN 保持数值尺度一致
            nn.BatchNorm2d(out_channels)
        )
    return None  # 维度完全一致，shortcut 走纯恒等连接，不引入任何参数
```

## 02.6 Global Average Pooling 分类头

==NiN章节里有详细讲解==

### 对比 VGG 的 Flatten + 三层 FC

在 AlexNet 和 VGG 的时代，分类头的标准做法是把最后一个卷积层的特征图 Flatten 成一个长向量，然后接几个全连接层：

```css
VGG-16 的分类头（以 224×224 输入为例）：
  最后卷积输出：[B, 512, 7, 7]
  → Flatten：  [B, 25088]
  → FC(25088 → 4096) + ReLU + Dropout(0.5)   ← 约 1 亿参数！
  → FC(4096  → 4096) + ReLU + Dropout(0.5)   ← 约 1600 万参数
  → FC(4096  → 1000)                         ← 约 400 万参数
  全连接部分共约 1.24 亿参数，占 VGG-16 总参数（1.38 亿）的 90%！
```

这种设计有两个根本性的问题：第一，绝大多数参数都堆在分类头，卷积层学到的丰富特征被强行压缩到一个固定维度的向量，信息利用率低；第二，Flatten 的操作要求输入必须是固定的空间尺寸（224×224），无法处理任意分辨率的图像。

ResNet 用 Global Average Pooling（GAP）彻底解决了这两个问题：

```css
ResNet-50 的分类头：
  最后卷积输出：[B, 2048, 7, 7]
  → AdaptiveAvgPool2d(1, 1)：[B, 2048, 1, 1]   ← 每个通道取全局平均
  → Flatten：                [B, 2048]
  → FC(2048 → num_classes)                      ← 仅约 200 万参数（ImageNet 1000类）
```

分类头参数量从 1.24 亿降到了 200 万，减少了 **60 倍**，同时性能反而更好。

### AdaptiveAvgPool2d 的原理与优势

`nn.AdaptiveAvgPool2d((1, 1))` 的语义是：无论输入的空间尺寸是多少，都将其压缩到 $1 \times 1$==也就是每一层通道压缩成一个表示数字==。对每个通道 $c$，输出就是该通道所有空间位置的平均值：

$$\text{output}[b, c, 0, 0] = \frac{1}{H \times W} \sum_{h=1}^{H} \sum_{w=1}^{W} \text{input}[b, c, h, w]$$

**为什么用平均而不是最大？** MaxPool 只保留最强激活，对于分类这种需要感知图像整体语义的任务，平均值能聚合整个特征图的信息，比只取最大值更能代表"这张图整体属于哪一类"。

```python
class ClassificationHead(nn.Module):
    """
    ResNet 的分类头：Global Average Pooling + 单层全连接。

    相比 VGG 的 Flatten + 三层 FC，参数量减少约 60 倍，
    且对任意输入空间尺寸均有效（Adaptive 池化的功劳）。

    Args:
        in_features: 全连接层的输入维度，等于最后一个卷积层的输出通道数
                     （ResNet-18/34 为 512，ResNet-50/101/152 为 2048）。
        num_classes: 分类任务的类别数量。
    """

    def __init__(self, in_features: int, num_classes: int):
        super(ClassificationHead, self).__init__()

        # 将任意空间尺寸的特征图压缩到 1×1，每通道取全局平均
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 单层全连接：完成从特征到类别概率的映射
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)        # [B, C, H, W] → [B, C, 1, 1]
        x = torch.flatten(x, 1)   # [B, C, 1, 1] → [B, C]
        x = self.fc(x)             # [B, C]        → [B, num_classes]
        return x
```

# 03 组装：从组件到完整网络

## 03.1 `_make_layer` 的设计思想

不用 `_make_layer`，手写 ResNet-50 的四个 Stage 是这样的：

```css
# 不用 _make_layer 的写法，全部手动展开
self.layer1_block1 = Bottleneck(64,  64,  stride=1, downsample=...)
self.layer1_block2 = Bottleneck(256, 64,  stride=1, downsample=None)
self.layer1_block3 = Bottleneck(256, 64,  stride=1, downsample=None)

self.layer2_block1 = Bottleneck(256, 128, stride=2, downsample=...)
self.layer2_block2 = Bottleneck(512, 128, stride=1, downsample=None)
self.layer2_block3 = Bottleneck(512, 128, stride=1, downsample=None)
self.layer2_block4 = Bottleneck(512, 128, stride=1, downsample=None)

# layer3 有 23 个 block...
# layer4 有 3 个 block...
```

ResNet-101 的 layer3 有 23 个 block，全手写要写 23 行几乎一样的代码。而且每个版本（18/34/50/101/152）的 block 数量都不一样，维护起来是灾难。`_make_layer` 的本质是把这种**有规律的重复结构**抽象成一个函数，用循环代替手写重复。



每个 Stage，内部有一个非常清晰的模式：

```css
Stage N：
  Block 1（特殊）：改变通道数 + 改变空间尺寸，需要 downsample
  Block 2（普通）：维度完全不变，直接恒等 shortcut
  Block 3（普通）：维度完全不变，直接恒等 shortcut
  ...
  Block n（普通）：维度完全不变，直接恒等 shortcut
```

**一个特殊的首 Block + 若干个完全相同的普通 Block**，这个模式在四个 Stage 里重复出现，只是 `planes`、`num_blocks`、`stride` 三个参数不同。`_make_layer` 就是对这个模式的直接抽象。



`_make_layer` 内部维护一个状态变量 `self.in_channels`，记录当前时刻输入到下一个 block 的通道数。这个变量在整个网络构建过程中不断更新：

```css
初始：self.in_channels = 64（Stem 的输出通道数）

Stage 1（planes=64）：
    首 block 输入 64，输出 64×exp
    self.in_channels 更新为 64×exp
    后续 block 均以 64×exp 为输入输出

Stage 2（planes=128）：
    首 block 输入 64×exp，输出 128×exp（需要 Downsample）
    self.in_channels 更新为 128×exp
    ...
```



`_make_layer` 的核心逻辑就是区分这两种情况：

- 为首 block 计算是否需要 downsample，如果需要则构建并传入
- 在首 block 创建完成后立即更新 `self.in_channels`
- 后续 block 以统一的 `self.in_channels` 为输入，无需 downsample，循环追加即可

```python
def _make_layer(
    self,
    block: type,
    planes: int,
    num_blocks: int,
    stride: int
) -> nn.Sequential:
    """
    构建一个完整的 Stage，包含 num_blocks 个残差块。

    设计上区分两类 block：
      · 第 1 个 block：可能需要下采样（stride≠1 或通道数变化），需要 downsample 投影。
      · 第 2~n 个 block：维度完全一致，shortcut 走恒等连接，无需 downsample。

    Args:
        block:      残差块类型，BasicBlock 或 Bottleneck。
        planes:     本 Stage 的基础通道数（实际输出 = planes × block.expansion）。
        num_blocks: 本 Stage 包含的残差块数量。
        stride:     首 block 的步长（Stage 2/3/4 的首 block 用 stride=2 做下采样）。

    Returns:
        nn.Sequential：由 num_blocks 个残差块串联而成的 Stage。

    Side Effects:
        更新 self.in_channels 为本 Stage 的输出通道数，
        供下一个 _make_layer 调用时使用。
    """
    # 计算本 Stage 的输出通道数
    out_channels = planes * block.expansion

    # ── 判断是否需要 shortcut 投影 ────────────────────────────────────
    # 两个触发条件（任一满足即需要投影）：
    #   1. stride != 1：需要空间下采样，shortcut 要跟主路径保持相同的空间尺寸
    #   2. self.in_channels != out_channels：通道数变化，维度不匹配无法直接相加
    downsample = None
    if stride != 1 or self.in_channels != out_channels:
        downsample = nn.Sequential(
            nn.Conv2d(
                self.in_channels, out_channels,
                kernel_size=1, stride=stride,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )

    layers = []

    # ── 创建首 block（可能含 downsample）────────────────────────────────
    layers.append(
        block(self.in_channels, planes, stride=stride, downsample=downsample)
    )
    # 首 block 创建完毕后，更新 in_channels 为本 Stage 的输出通道数
    # 这个更新必须在创建后续 block 之前完成
    self.in_channels = out_channels

    # ── 创建后续 block（维度一致，无需 downsample）──────────────────────
    for _ in range(1, num_blocks):
        # stride 默认为 1（不改变空间尺寸）
        # downsample 默认为 None（in_channels == out_channels，直接恒等）
        layers.append(block(self.in_channels, planes))

    return nn.Sequential(*layers)
```

**ResNet-18/34/50/101/152 的 layers 配置对比:**

|    版本    | Block 类型 | Stage 1 | Stage 2 | Stage 3 | Stage 4 | 总层数（含 Stem+FC） |
| :--------: | :--------: | :-----: | :-----: | :-----: | :-----: | :------------------: |
| ResNet-18  | BasicBlock |    2    |    2    |    2    |    2    |          18          |
| ResNet-34  | BasicBlock |    3    |    4    |    6    |    3    |          34          |
| ResNet-50  | Bottleneck |    3    |    4    |    6    |    3    |          50          |
| ResNet-101 | Bottleneck |    3    |    4    |   23    |    3    |         101          |
| ResNet-152 | Bottleneck |    3    |    8    |   36    |    3    |         152          |

注意 ResNet-50 与 ResNet-34 的 layers 配置 `[3, 4, 6, 3]` 完全相同，只是把 Block 类型从 BasicBlock 换成了 Bottleneck，层数就从 34 跳到了 50（因为每个 Bottleneck 有 3 个卷积层而不是 2 个）

## 03.2 完整 ResNet 类

代码分成五个部分，逻辑是自上而下的依赖关系：

**Part 1 BasicBlock** → **Part 2 Bottleneck**：两个 Block 各自独立定义，都有 `expansion` 类属性，这是让 ResNet 类可以用同一套代码同时支持两种 Block 的关键。

**Part 3 ResNet 类**：依赖上面两个 Block。`__init__` 里的四次 `_make_layer` 调用通过 `self.in_channels` 这个状态变量串联起来，每次调用结束后自动更新通道数，下一次调用直接读取，不需要手动传递。

**Part 4 工厂函数**：五个函数只是对 `ResNet(block, layers, ...)` 的封装，差异只在 `block` 类型和 `layers` 列表两个参数，对照着看五个版本的配置差异一目了然。

**Part 5 验证**：跑通之后会输出参数量表、任意尺寸兼容性验证、自定义类别数验证三部分，可以用来确认代码没有问题。

```python
"""
ResNet 完整实现
支持：ResNet-18 / 34 / 50 / 101 / 152

参考论文：
  - He et al., Deep Residual Learning for Image Recognition, CVPR 2016
  - He et al., Identity Mappings in Deep Residual Networks, ECCV 2016
"""

import torch
import torch.nn as nn
from typing import List, Optional, Type, Union


# ──────────────────────────────────────────────────────────────────────────────
# Part 1：BasicBlock（用于 ResNet-18 / 34）
# ──────────────────────────────────────────────────────────────────────────────

class BasicBlock(nn.Module):
    """
    ResNet-18 / ResNet-34 使用的基础残差块。

    结构：
        主路径：Conv(3×3) → BN → ReLU → Conv(3×3) → BN
        shortcut：恒等连接 或 1×1 Conv + BN（维度不匹配时）
        输出：主路径 + shortcut → ReLU

    参数量（通道数为 C 时）：
        2 × (3×3×C×C) = 18C²

    Attributes:
        expansion (int): 输出通道相对于 planes 的倍数。
                         BasicBlock 不扩张通道，固定为 1。
    """

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ) -> None:
        """
        Args:
            in_channels: 输入特征图的通道数。
            planes:      本 Block 的基础通道数。
                         输出通道数 = planes × expansion = planes。
            stride:      第一个卷积的步长。
                         stride=1：不改变空间尺寸（Stage 内部的普通 Block）。
                         stride=2：空间尺寸减半（跨 Stage 的首 Block）。
            downsample:  shortcut 的投影层，由 _make_layer 构建后传入。
                         None 表示 shortcut 直接走恒等连接。
        """
        super(BasicBlock, self).__init__()

        # ── 主路径 ────────────────────────────────────────────────────────
        # 第一个卷积：负责通道变换（in_channels → planes）和可能的空间下采样（stride）
        self.conv1 = nn.Conv2d(
            in_channels, planes,
            kernel_size=3, stride=stride, padding=1,
            bias=False      # 紧跟 BN，BN 的 beta 等效于 bias，此处不需要
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)   # inplace=True：原地修改，节省显存

        # 第二个卷积：保持通道数和空间尺寸不变（stride 固定为 1）
        self.conv2 = nn.Conv2d(
            planes, planes,
            kernel_size=3, stride=1, padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        # 注意：这里没有 ReLU，残差相加之后统一激活，避免加法前提前截断负值

        # ── shortcut 路径 ─────────────────────────────────────────────────
        self.downsample = downsample    # None → 恒等；非 None → 1×1 Conv + BN 投影

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x    # 保存输入，用于后续残差相加

        # 主路径前向
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)     # 注意：这里不接 ReLU

        # shortcut 前向：维度不匹配时做投影，否则直接用原始输入
        if self.downsample is not None:
            identity = self.downsample(x)   # 对原始 x 投影，不是对 out 投影！

        # 残差相加 + 统一激活
        out = out + identity    # F(x) + x，核心残差操作
        out = self.relu(out)    # 相加后统一做非线性激活

        return out


# ──────────────────────────────────────────────────────────────────────────────
# Part 2：Bottleneck（用于 ResNet-50 / 101 / 152）
# ──────────────────────────────────────────────────────────────────────────────

class Bottleneck(nn.Module):
    """
    ResNet-50 / ResNet-101 / ResNet-152 使用的瓶颈残差块。

    结构：
        主路径：Conv(1×1，降维) → BN → ReLU
              → Conv(3×3，空间卷积) → BN → ReLU
              → Conv(1×1，升维) → BN
        shortcut：恒等连接 或 1×1 Conv + BN（维度不匹配时）
        输出：主路径 + shortcut → ReLU

    设计动机：
        在 Stage 通道数较大时（如 256、512），直接用两个 3×3 卷积参数量极大。
        Bottleneck 先用 1×1 将通道数压缩 4 倍，在低维空间做 3×3 卷积，
        再用 1×1 升维还原，大幅降低参数量的同时保持表达能力。

    参数量（瓶颈通道为 C，输出通道为 4C 时）：
        1×1×(4C)×C + 3×3×C×C + 1×1×C×(4C) = 17C²
        远小于同等通道的 BasicBlock：18×(4C)² = 288C²

    Attributes:
        expansion (int): 输出通道相对于 planes 的倍数，固定为 4。
                         即 planes=64 时，输出通道为 256。
    """

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ) -> None:
        """
        Args:
            in_channels: 输入通道数（通常是上一个 Block 的输出，即 planes × 4）。
            planes:      瓶颈通道数（中间 3×3 卷积操作的通道维度）。
                         输出通道数 = planes × expansion = planes × 4。
            stride:      下采样步长，放在 3×3 卷积上（不放在第一个 1×1 上）。
                         原因：让下采样发生在有空间感知的层，信息损失更少。
            downsample:  shortcut 投影层，维度不匹配时使用。
        """
        super(Bottleneck, self).__init__()

        # ── 1×1 降维 ──────────────────────────────────────────────────────
        # 将通道从 in_channels 压缩到 planes（如 256 → 64）
        # kernel_size=1：不涉及空间操作，纯粹做通道维度的线性组合
        self.conv1 = nn.Conv2d(in_channels, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)

        # ── 3×3 空间卷积 ──────────────────────────────────────────────────
        # 在低维瓶颈（planes 通道）上做空间特征提取，计算量大幅降低
        # stride 放在这里：让下采样和空间特征提取同时发生，比放在 1×1 上信息损失更少
        self.conv2 = nn.Conv2d(
            planes, planes,
            kernel_size=3, stride=stride, padding=1,
            bias=False
        )
        self.bn2   = nn.BatchNorm2d(planes)

        # ── 1×1 升维 ──────────────────────────────────────────────────────
        # 将通道从 planes 扩张到 planes × expansion（如 64 → 256）
        # 目的：与 shortcut 对齐，使得残差相加可以进行
        # 注意：末尾不接 ReLU，残差相加之后统一激活
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion,
            kernel_size=1, bias=False
        )
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)

        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # 主路径：降维 → 空间卷积 → 升维
        out = self.relu(self.bn1(self.conv1(x)))    # 1×1 降维 → BN → ReLU
        out = self.relu(self.bn2(self.conv2(out)))  # 3×3 卷积 → BN → ReLU
        out = self.bn3(self.conv3(out))             # 1×1 升维 → BN（无 ReLU）

        # shortcut 投影（维度不匹配时）
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差相加 + 统一激活
        out = out + identity
        out = self.relu(out)

        return out


# ──────────────────────────────────────────────────────────────────────────────
# Part 3：完整 ResNet
# ──────────────────────────────────────────────────────────────────────────────

class ResNet(nn.Module):
    """
    通用 ResNet，通过 block 类型和 layers 配置支持所有主流版本。

    网络结构：
        Stem（Conv 7×7 + BN + ReLU + MaxPool）
        → Stage 1 / 2 / 3 / 4（各由若干残差块构成）
        → 分类头（AdaptiveAvgPool2d + Linear）

    版本对应关系：
        ResNet-18 ：BasicBlock,  [2, 2, 2,  2]
        ResNet-34 ：BasicBlock,  [3, 4, 6,  3]
        ResNet-50 ：Bottleneck,  [3, 4, 6,  3]
        ResNet-101：Bottleneck,  [3, 4, 23, 3]
        ResNet-152：Bottleneck,  [3, 8, 36, 3]

    Args:
        block:               残差块类型，BasicBlock 或 Bottleneck。
        layers:              长度为 4 的列表，每个元素是对应 Stage 的 Block 数量。
        num_classes:         分类类别数，默认 1000（ImageNet）。
        zero_init_residual:  是否将每个残差块末尾 BN 的 gamma 初始化为 0。
                             效果：训练初期网络等价于更浅的结构，收敛更稳定。
                             实验结果：在 ImageNet 上可提升约 0.2~0.3% Top-1 精度。
    """

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False
    ) -> None:
        super(ResNet, self).__init__()

        # self.in_channels：状态变量，记录当前输入到下一个 Stage 的通道数。
        # 初始为 64（Stem 输出通道数），每次调用 _make_layer 后自动更新。
        # 四次调用之间通过这个变量隐式传递通道信息，无需手动指定每个 Stage 的输入通道。
        self.in_channels = 64

        # ── Stem：入口模块 ────────────────────────────────────────────────
        # 目的：对高分辨率输入快速下采样，降低后续 Stage 的计算量
        # 两步下采样：Conv(stride=2) 224→112，MaxPool(stride=2) 112→56
        self.conv1 = nn.Conv2d(
            3, 64,
            kernel_size=7, stride=2, padding=3,
            bias=False      # 后接 BN，bias 冗余
            # padding=3：保证输出尺寸精确减半 (224+2×3-7)/2+1=112
        )
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1
            # padding=1：保证输出尺寸精确减半 (112+2×1-3)/2+1=56，不丢边缘信息
        )

        # ── 四个 Stage ────────────────────────────────────────────────────
        # Stage 1：stride=1，Stem 已经做过两次下采样，这里不再缩小尺寸
        # 输出：[B, 64×exp,  56, 56]（BasicBlock→64，Bottleneck→256）
        self.layer1 = self._make_layer(block, planes=64,  num_blocks=layers[0], stride=1)

        # Stage 2：stride=2，56×56 → 28×28
        # 输出：[B, 128×exp, 28, 28]
        self.layer2 = self._make_layer(block, planes=128, num_blocks=layers[1], stride=2)

        # Stage 3：stride=2，28×28 → 14×14
        # 输出：[B, 256×exp, 14, 14]
        self.layer3 = self._make_layer(block, planes=256, num_blocks=layers[2], stride=2)

        # Stage 4：stride=2，14×14 → 7×7
        # 输出：[B, 512×exp,  7,  7]
        self.layer4 = self._make_layer(block, planes=512, num_blocks=layers[3], stride=2)

        # ── 分类头 ────────────────────────────────────────────────────────
        # AdaptiveAvgPool2d：将任意空间尺寸压缩到 1×1，每通道取全局平均值
        # 好处：对任意输入分辨率兼容，参数量远小于 VGG 式的多层 FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # FC：完成最终的类别预测
        # in_features = 512 × expansion（BasicBlock→512，Bottleneck→2048）
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # ── 权重初始化 ────────────────────────────────────────────────────
        self._init_weights(zero_init_residual)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        num_blocks: int,
        stride: int
    ) -> nn.Sequential:
        """
        构建一个完整的 Stage，包含 num_blocks 个残差块。

        Stage 内部的结构规律：
            Block 1（特殊）：可能需要改变通道数和空间尺寸，shortcut 需要投影。
            Block 2~n（普通）：输入输出维度完全一致，shortcut 直接恒等连接。

        Args:
            block:      残差块类型，BasicBlock 或 Bottleneck。
            planes:     本 Stage 的基础通道数。
            num_blocks: 本 Stage 的 Block 总数量。
            stride:     首 Block 的步长（Stage 1 传 1，Stage 2/3/4 传 2）。

        Returns:
            nn.Sequential：由 num_blocks 个残差块串联构成的 Stage。

        Side Effects:
            更新 self.in_channels 为本 Stage 的输出通道数，
            供下一次 _make_layer 调用时读取。
        """
        # 计算本 Stage 的输出通道数
        # BasicBlock.expansion=1 → out_channels = planes
        # Bottleneck.expansion=4 → out_channels = planes × 4
        out_channels = planes * block.expansion

        # ── 判断首 Block 的 shortcut 是否需要投影 ─────────────────────────
        # 触发条件（任一满足即需要投影）：
        #   stride != 1           → 空间尺寸要变，shortcut 必须同步下采样
        #   in_channels != out_channels → 通道数要变，shortcut 必须同步扩张
        # 对于 Stage 2/3/4 的首 Block，两个条件通常同时满足
        # 对于 Stage 1 的首 Block（Bottleneck），只有通道数条件满足（64 ≠ 256）
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, out_channels,
                    kernel_size=1, stride=stride,
                    bias=False          # 后接 BN，bias 冗余
                    # kernel_size=1：不改变空间感受野，只做通道线性投影
                    # stride：与主路径的下采样步长保持一致，保证空间尺寸对齐
                ),
                nn.BatchNorm2d(out_channels)
                # BN：与主路径末尾的 BN 对齐数值尺度，相加时两条路径贡献均衡
            )

        layers = []

        # ── 首 Block：特殊，可能有 downsample 和 stride=2 ─────────────────
        layers.append(
            block(
                in_channels=self.in_channels,
                planes=planes,
                stride=stride,          # 首 Block 负责下采样（如果需要）
                downsample=downsample   # 传入投影层（可能为 None）
            )
        )

        # 首 Block 创建完毕，立即更新状态变量
        # 后续 Block 的输入通道数已经是 out_channels 了
        self.in_channels = out_channels

        # ── 后续 Block：普通，维度一致，无需 downsample ────────────────────
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    in_channels=self.in_channels,   # 此时已等于 out_channels
                    planes=planes
                    # stride 默认 1：不改变空间尺寸
                    # downsample 默认 None：in_channels == out_channels，走恒等 shortcut
                )
            )
            # 不需要再更新 self.in_channels，普通 Block 输入输出通道数相同

        # 将所有 Block 打包成 Sequential，前向传播时数据依次流过每个 Block
        return nn.Sequential(*layers)

    def _init_weights(self, zero_init_residual: bool) -> None:
        """
        初始化网络中所有层的权重。

        策略：
            Conv2d      → Kaiming 正态初始化（专为 ReLU 设计，保持各层输出方差稳定）
            BatchNorm2d → gamma=1，beta=0（标准初始化，不干扰初始特征分布）
            zero_init_residual → 每个残差块末尾 BN 的 gamma 初始化为 0
                                  使训练初期残差分支输出为 0，网络行为等价于更浅的结构，
                                  随训练进行 gamma 从 0 增大，网络从"浅"逐渐"生长"

        Args:
            zero_init_residual: 是否开启残差块末尾 BN gamma=0 的初始化技巧。
        """
        for m in self.modules():    # 递归遍历网络中的所有子模块
            if isinstance(m, nn.Conv2d):
                # Kaiming 初始化：
                #   mode='fan_out'：以输出神经元数为基准，适合分析前向传播的方差
                #   nonlinearity='relu'：对应 ReLU 的增益系数 sqrt(2)，防止方差衰减
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)  # gamma=1：初始不缩放特征
                nn.init.constant_(m.bias,   0)  # beta=0 ：初始不偏移特征

        # zero_init_residual 技巧（可选）：
        # 将每个残差块最后一个 BN 的 gamma 设为 0
        # 效果：训练初期每个残差块的主路径输出乘以 0 等于 0，
        #       整个网络退化为只有 shortcut 的极浅网络，行为非常稳定
        # 随训练进行，gamma 从 0 开始学习增大，残差分支逐渐参与贡献
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # 最后一个 BN 是 bn3
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # 最后一个 BN 是 bn2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Stem ──────────────────────────────────────────────────────────
        x = self.conv1(x)       # [B,   3, 224, 224] → [B, 64, 112, 112]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # [B,  64, 112, 112] → [B, 64,  56,  56]

        # ── 四个 Stage ────────────────────────────────────────────────────
        x = self.layer1(x)      # [B,    64, 56, 56] → [B,  64×exp, 56, 56]
        x = self.layer2(x)      # [B, 64×exp, 56,56] → [B, 128×exp, 28, 28]
        x = self.layer3(x)      # [B,128×exp, 28,28] → [B, 256×exp, 14, 14]
        x = self.layer4(x)      # [B,256×exp, 14,14] → [B, 512×exp,  7,  7]

        # ── 分类头 ────────────────────────────────────────────────────────
        x = self.avgpool(x)         # [B, 512×exp, 7, 7] → [B, 512×exp, 1, 1]
        x = torch.flatten(x, 1)    # [B, 512×exp, 1, 1] → [B, 512×exp]
        x = self.fc(x)              # [B, 512×exp]       → [B, num_classes]

        return x


# ──────────────────────────────────────────────────────────────────────────────
# Part 4：工厂函数（根据版本名称直接创建对应模型）
# ──────────────────────────────────────────────────────────────────────────────

def resnet18(num_classes: int = 1000, zero_init_residual: bool = False) -> ResNet:
    """
    ResNet-18。

    配置：BasicBlock × [2, 2, 2, 2]，共 18 层（含 Stem 和 FC）
    特点：参数量最小（11.7M），推理速度最快，适合边缘设备或快速实验。
    """
    return ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=num_classes,
        zero_init_residual=zero_init_residual
    )


def resnet34(num_classes: int = 1000, zero_init_residual: bool = False) -> ResNet:
    """
    ResNet-34。

    配置：BasicBlock × [3, 4, 6, 3]，共 34 层
    特点：比 ResNet-18 更深，仍然使用 BasicBlock，精度有提升（约 73% Top-1）。
    """
    return ResNet(
        block=BasicBlock,
        layers=[3, 4, 6, 3],
        num_classes=num_classes,
        zero_init_residual=zero_init_residual
    )


def resnet50(num_classes: int = 1000, zero_init_residual: bool = False) -> ResNet:
    """
    ResNet-50。

    配置：Bottleneck × [3, 4, 6, 3]，共 50 层
    特点：最常用的版本，切换到 Bottleneck 后精度大幅提升（约 76% Top-1），
          参数量（25.6M）与 ResNet-34（21.8M）相近但表达能力更强。
    注意：layers 配置与 ResNet-34 完全相同，仅 Block 类型不同。
    """
    return ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        num_classes=num_classes,
        zero_init_residual=zero_init_residual
    )


def resnet101(num_classes: int = 1000, zero_init_residual: bool = False) -> ResNet:
    """
    ResNet-101。

    配置：Bottleneck × [3, 4, 23, 3]，共 101 层
    特点：Stage 3 增加到 23 个 Bottleneck，精度进一步提升（约 77% Top-1），
          参数量 44.5M，计算开销明显增大。
    """
    return ResNet(
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        num_classes=num_classes,
        zero_init_residual=zero_init_residual
    )


def resnet152(num_classes: int = 1000, zero_init_residual: bool = False) -> ResNet:
    """
    ResNet-152。

    配置：Bottleneck × [3, 8, 36, 3]，共 152 层
    特点：最重的标准版本，Stage 2 和 Stage 3 大幅加深，精度最高（约 78% Top-1），
          参数量 60.2M，推理速度最慢。
    """
    return ResNet(
        block=Bottleneck,
        layers=[3, 8, 36, 3],
        num_classes=num_classes,
        zero_init_residual=zero_init_residual
    )


# ──────────────────────────────────────────────────────────────────────────────
# Part 5：验证与测试
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    print("=" * 65)
    print(f"{'版本':<12} {'输出形状':<28} {'参数量':>10}")
    print("=" * 65)

    factory_fns = [
        ('ResNet-18',  resnet18),
        ('ResNet-34',  resnet34),
        ('ResNet-50',  resnet50),
        ('ResNet-101', resnet101),
        ('ResNet-152', resnet152),
    ]

    for name, fn in factory_fns:
        model = fn(num_classes=1000)
        model.eval()

        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)

        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"{name:<12} {str(out.shape):<28} {total_params:>8.1f}M")

    print("=" * 65)

    # ── 验证对任意输入尺寸的兼容性 ─────────────────────────────────────
    print("\n验证任意输入尺寸兼容性（ResNet-50）：")
    model = resnet50(num_classes=10)
    model.eval()

    for size in [224, 320, 448]:
        x = torch.randn(1, 3, size, size)
        with torch.no_grad():
            out = model(x)
        print(f"  输入 {size}×{size} → 输出 {out.shape}")

    # ── 验证自定义类别数 ─────────────────────────────────────────────────
    print("\n验证自定义类别数：")
    for nc in [10, 100, 200]:
        model = resnet50(num_classes=nc)
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        print(f"  num_classes={nc:<4} → 输出 {out.shape}")
```

# 04 设计思想延伸

## 04.1 ResNeXt：用分组卷积改进 Bottleneck

2017 年，恺明大佬团队提出了 ResNeXt，在 Bottleneck 的基础上引入了"基数（Cardinality）"的概念。核心思路是：与其加深网络（更多层）或加宽网络（更多通道），不如增加变换的**并行路径数**。

### Cardinality 的概念

ResNeXt 把 Bottleneck 中间的 3×3 卷积替换为**分组卷积（Group Convolution）**：

```css
标准 Bottleneck（ResNet-50 中的一个块）：
  1×1(256→64) → 3×3(64→64) → 1×1(64→256)

ResNeXt Bottleneck（cardinality=32, base_width=4）：
  1×1(256→128) → 3×3(128→128, groups=32) → 1×1(128→256)
  ↑ 32 个并行的 4 通道分组，每组独立做 3×3 卷积
```

分组卷积把 128 个通道分成 32 组，每组 4 个通道独立做 3×3 卷积，然后把 32 组的结果拼接起来。这等价于 32 个并行的小 Bottleneck，每个只处理 4 通道，最后把所有结果融合。这种多路并行的聚合方式在相同参数量下，表达能力比单一路径更强

==分组卷积不是在"隔离特征"，而是在 1×1 全通道混合的前提下，让 3×3 空间提取阶段变得更轻量、更多样==

### 与 Bottleneck 的代码差异

ResNeXt 与 ResNet 的代码差异极小，只需在 Bottleneck 的 3×3 卷积上增加 `groups` 参数即可：

```python
class ResNeXtBottleneck(nn.Module):
    """
    ResNeXt 的瓶颈块。
    与标准 Bottleneck 的唯一区别：3×3 卷积使用分组卷积（groups > 1）。

    Args:
        in_channels: 输入通道数。
        planes:      每组的基础通道数（实际中间通道数 = planes × groups_width）。
        stride:      下采样步长。
        downsample:  shortcut 投影层。
        groups:      分组数（Cardinality），论文推荐 32。
        base_width:  每组的通道宽度，论文推荐 4（中间通道总数 = groups × base_width）。
    """

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module = None,
        groups: int = 32,        # Cardinality，并行路径数
        base_width: int = 4      # 每组的通道宽度
    ):
        super(ResNeXtBottleneck, self).__init__()

        # 计算实际中间通道数（随 planes 缩放，保持各层比例一致）
        width = int(planes * (base_width / 64.)) * groups  # 如 planes=64 时 width=128

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(width)

        # 核心区别：groups 参数将通道分组，实现多路并行卷积
        self.conv2 = nn.Conv2d(
            width, width,
            kernel_size=3, stride=stride, padding=1,
            groups=groups,  # ← 这里是与标准 Bottleneck 唯一的差异
            bias=False
        )
        self.bn2   = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)

        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


# torchvision 中的调用方式
import torchvision.models as models
model_resnext = models.resnext50_32x4d(weights=None)
# 参数命名：resnext50_32x4d = ResNeXt-50，cardinality=32，base_width=4
```

## 04.2 残差思想的继承

### Vision Transformer：`x + MSA(LN(x))` 的本质

2020 年提出的 Vision Transformer（ViT）看起来与 ResNet 完全不同：没有卷积，用自注意力机制处理图像块序列。但如果仔细观察每个 Transformer Block 的结构：

```
ViT 的一个 Transformer Block：

x → LayerNorm → Multi-Head Self-Attention → + x   ← 残差连接
                                            ↓
  → LayerNorm → MLP(Feed Forward)        → + x   ← 残差连接
```

每个子层（自注意力和 MLP）的输出都被加回了输入，这与 ResNet 的 $H(x) = F(x) + x$ 完全同构。唯一的区别是 ViT 把 BN 换成了 Layer Normalization（因为 BN 在序列数据上效果不好），并且采用了 Pre-activation 的风格（先做 LN，再做子层变换），这与 ResNet v2 的 Pre-activation Block 的思路完全一致



可以说，ViT 在架构哲学上是残差连接思想的直接继承者。如果没有残差连接，12 层（ViT-Base）或 24 层（ViT-Large）的 Transformer 根本无法稳定训练

### BERT/GPT：为什么超深 Transformer 也离不开残差

BERT 有 12 层 Transformer（BERT-Base）和 24 层（BERT-Large），GPT-3 有 96 层。这些模型的每一层都使用了与上述完全相同的残差结构

如果 GPT-3 的 96 层没有残差连接，梯度从输出层传回最浅层时，需要经过 96 层的雅可比矩阵连乘，几乎必然消失殆尽，模型根本无法训练。残差连接提供的那条"梯度高速公路"是使超深 Transformer 得以存在的结构基础

### 总结：残差是"让优化器只学增量"的通用哲学

从退化问题出发，到改善梯度流，到 ViT、BERT、GPT 的结构基础，残差连接的核心思想可以用一句话概括：

> **不要让网络从零开始学习一个完整的变换，而是让它在已有信息的基础上学习需要做出的修正（增量）。** 当最优修正是"什么都不做"时，网络只需把权重推向零，这在优化上是最容易的事；当需要做出改变时，网络学到的每一点都是相对于恒等映射的真实提升

这个哲学不仅适用于图像、语言，它是深度学习中对"深层网络优化困难"这一根本问题的普遍解法

