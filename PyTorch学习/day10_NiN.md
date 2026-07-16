# 01 NiN背景

## 01.1 问题1

在 AlexNet 和 VGG 的时代，卷积层的工作方式是这样的：用一个线性卷积核在输入特征图上滑动，做内积运算，然后接一个 ReLU 激活。这个内积运算的本质还是一个**线性变换**

```lua
传统卷积层的计算过程：
输入局部区域（patch）→ 与卷积核做内积 → 得到一个标量 → ReLU → 输出一个值

数学表达：
output = ReLU(W · x + b)

其中 W · x 是线性操作，整个表达能力受限于线性函数
```

这有一个根本性的局限：**单个卷积核只能提取线性可分的特征**。如果某个局部区域的特征需要非线性组合才能被识别，一层卷积是做不到的，必须靠多层卷积的堆叠来间接实现

## 01.2 问题2

全连接层的参数爆炸问题：VGG-16 的卷积部分输出 `(B, 512, 7, 7)`，展平后是 `512×7×7 = 25088` 个数字，第一个全连接层参数量 = `25088 × 4096 ≈ 1 亿`。

**参数爆炸的根源：空间维度 7×7 没有被消掉，直接被打平进了输入向量。** VGG-16 约 1.38 亿参数中，87% 在全连接层。

# 02 MLPConv

NiN 的核心模块叫做 **MLPConv**，全称是 MLP Convolutional Layer。理解它的关键是：普通卷积核在一个局部窗口上做的是线性变换，而 MLPConv 在同一个局部窗口上做的是一个多层感知机（MLP）的非线性变换

```css
普通卷积（对每个局部 patch 的操作）：
patch → [线性变换] → 一个输出值

MLPConv（对同一个 patch 的操作）：
patch → [线性变换] → ReLU → [线性变换] → ReLU → 多个输出值
```

==这里有一个非常巧妙的等价变换：**在一个局部 patch 上做 MLP，等价于先做一次普通卷积，再接若干个 1×1 卷积**==



**这个"在局部窗口上运行 MLP"，和 1×1 卷积是什么关系？**

将这个非线性变换展开来看：

第一步：普通 3×3 卷积。输入 64 通道，用 96 个 3×3×64 的卷积核，输出 96 通道。此时对每个空间位置，有了一个 96 维的向量

```css
输入：(H, W, 64)
↓ 3×3 卷积，96个核
输出：(H, W, 96)    ← 每个位置是一个 96 维向量，代表 96 种局部特征的响应
```

第二步：问题来了——这 96 维向量需要进一步非线性组合。NiN 的解法：**在每个空间位置，对这 96 维向量再做一次全连接变换（线性）+ ReLU（非线性），再做一次，得到更深层的特征**

```css
某个位置的 96 维向量：
[v1, v2, v3, ..., v96]
↓ 全连接（96→96）+ ReLU
[u1, u2, u3, ..., u96]
↓ 全连接（96→96）+ ReLU
[w1, w2, w3, ..., w96]
```

这个"对每个位置的向量做全连接"，正好等价于 1×1 卷积：

```css
1×1 卷积核（大小是 1×1×96，共 96 个这样的核，输出 96 通道）：
对某个位置 (h, w) 的 96 维向量：
  第1个输出 = w1[0]×v1 + w1[1]×v2 + ... + w1[95]×v96  （第1个核的内积）
  第2个输出 = w2[0]×v1 + w2[1]×v2 + ... + w2[95]×v96  （第2个核的内积）
  ...
  第96个输出 = w96[0]×v1 + ...                          （第96个核的内积）

这正好是：把 96 维向量乘以一个 96×96 的权重矩阵，得到新的 96 维向量。
这就是全连接层！
一个全连接层本质上就是一个矩阵乘法。输入是一个向量，输出是另一个向量，中间的矩阵里每一格都是一个可学习的权重。
```

所以，在每个位置对 96 维向量做全连接 ≡ 对整张特征图做 1×1 卷积（这两种说法描述的是同一个操作）

# 03 全局平均池化

在 VGG 等网络中，卷积层输出的特征图要接入全连接层，必须先 Flatten 展开：

```css
VGG 最后的卷积输出：(B, 512, 7, 7)
Flatten 后：(B, 512×7×7) = (B, 25088)
第一个 FC：Linear(25088, 4096)  → 参数量：25088×4096 ≈ 1亿
```

NiN 完全抛弃了这个做法。它的思路是：**让最后一个 MLPConv 模块的输出通道数等于类别数，然后对每个通道求全局平均，直接得到类别分数**

> * 全连接层把特征图所有位置的所有通道拼在一起，再学习一个巨大的权重矩阵，这个权重矩阵是空间位置敏感的（它"知道"某个类别的特征应该出现在特征图的哪个区域）。这导致两个问题：参数爆炸 + 对空间位置过于敏感（泛化性差）
> * 全局平均池化相当于：对每个通道的所有空间位置的响应取平均，然后直接用这个平均响应作为该类别的置信度。它的含义更接近"全图中，属于第 i 类特征的平均激活强度"，这是一种更全局、更鲁棒的表示
> * 此外，全局平均池化没有参数，完全不会过拟合

## 03.1 做法

不把空间位置区分对待，而是把每个通道在所有空间位置上的响应平均起来，用这个平均值代表该通道的整体激活程度

> * 我的理解是每一层通道，其实都是在学习一种模式，那么，只要看这层通道的一个总体情况就可以表示这种模式，而不需要去考虑这层通道的特征图的空间位置信息。比如说这层通道是学习到了猫有眼睛，那么只要这一层能表达眼睛即可，而不需要去关注这个眼睛到底是在什么空间位置才表示是猫，所以用通道的平均值来表示即可

```css
输入：(B, 512, 7, 7)

全局平均池化对每个通道：
  通道 0 的平均值 = 对这个通道的 7×7=49 个值取平均 → 一个数字
  通道 1 的平均值 = 对这个通道的 49 个值取平均     → 一个数字
  ...
  通道 511 的平均值 = ...                            → 一个数字

输出：(B, 512)    ← 512 个数字，每个代表一个通道的"全图平均响应强度"
```

==**参数量 = 0。** 取平均不需要任何可学习的参数==

**NiN 更进一步：让最后一层的通道数直接等于类别数**

NiN 不是在 512 通道的特征图后面接全局平均池化，而是让最后一个 Block 的输出通道数直接等于类别数（比如 10）

```css
NiN 的最后一步：

Block4 输出：(B, 10, 4, 4)
  ↓ 全局平均池化
(B, 10, 1, 1)
  ↓ Flatten
(B, 10)   ← 直接就是 10 个类别的分数，不需要任何 FC 层！
```

每个通道对应一个类别，该通道在全图上的平均激活强度，就代表"这张图属于该类别的置信度"



```css
VGG 的做法（全连接层）：
  卷积输出 (B, 512, 7, 7)
  → Flatten → (B, 25088)
  → FC(25088, 4096) → 参数：102,760,448
  → FC(4096, 4096)  → 参数：16,777,216
  → FC(4096, 1000)  → 参数：4,096,000
  总计：约 1.24 亿参数

NiN 的做法（全局平均池化）：
  最后一个 Block 输出 (B, 10, 4, 4)
  → GlobalAvgPool → (B, 10, 1, 1)
  → Flatten → (B, 10)
  参数：0
```

**全局平均池化**的逻辑是：我不关心某个特征出现在图像的哪个位置，我只关心这个特征在整张图上有没有出现、出现得强不强。这就像看一篇文章判断主题，不需要记住每个词在第几行，只需要统计各类词的出现频率

这也解释了为什么全局平均池化对于平移更鲁棒——同一只猫，不管出现在图像左边还是右边，该类别对应的通道平均激活强度是差不多的。而全连接层因为记住了空间位置，对平移非常敏感

## 03.2 问题梳理

> * 把最后一层卷积的通道数设成 10，这样做有意义吗？训练能收敛吗？**

这才是 NiN 真正的洞察。

普通的思维是：卷积层负责提特征，全连接层负责分类，两者分工明确，不能混用。

NiN 打破了这个思维：**卷积层的最后一层，为什么不能直接承担分类的职责？** 我就让最后一个卷积块输出 10 个通道，然后用损失函数强迫训练过程让这 10 个通道分别学会响应 10 个类别。训练会自己把这件事做成。

所以整个逻辑链是：

```css
1. 卷积层输出几个通道，由你设计，没有限制
2. NiN 把最后一个卷积块的输出通道设计成 num_classes
3. 全局平均池化每个通道出一个数，输出恰好是 num_classes 个数字
4. 训练过程中损失函数会逼着这 num_classes 个通道各自学会对应一个类别
5. 全连接层因此完全不需要了
```

之前卡住的地方，其实是潜意识里觉得"卷积层不应该输出类别数那么少的通道"。但实际上没有这个限制，这只是一个设计选择。

```css
输入图片
(1, 3, 32, 32)
 ↑  ↑   ↑  ↑
 B  通道  H  W
    RGB

══════════════════════════════════════
Block 1 第一层：5×5 卷积，输出96通道，padding=2
══════════════════════════════════════

(1, 3, 32, 32)
    ↓ Conv2d(3→96, kernel=5, padding=2)
(1, 96, 32, 32)

尺寸计算：(32 - 5 + 2×2) / 1 + 1 = 32，空间尺寸不变

    ↓ ReLU

(1, 96, 32, 32)

──────────────────────────────────────
Block 1 第二层：1×1 卷积
──────────────────────────────────────

(1, 96, 32, 32)
    ↓ Conv2d(96→96, kernel=1)
(1, 96, 32, 32)

1×1 卷积不改变空间尺寸，只在通道维度做变换
每个位置的 96 个数字，重新线性组合成新的 96 个数字

    ↓ ReLU

(1, 96, 32, 32)

──────────────────────────────────────
Block 1 第三层：1×1 卷积
──────────────────────────────────────

(1, 96, 32, 32)
    ↓ Conv2d(96→96, kernel=1)
(1, 96, 32, 32)
    ↓ ReLU
(1, 96, 32, 32)

══════════════════════════════════════
MaxPool（32→16）
══════════════════════════════════════

(1, 96, 32, 32)
    ↓ MaxPool2d(kernel=3, stride=2, padding=1)
(1, 96, 16, 16)

池化只改变空间尺寸，通道数 96 不变

══════════════════════════════════════
Block 2 第一层：5×5 卷积，输出256通道
══════════════════════════════════════

(1, 96, 16, 16)
    ↓ Conv2d(96→256, kernel=5, padding=2)
(1, 256, 16, 16)

通道数从 96 变成 256，空间尺寸不变

    ↓ ReLU
(1, 256, 16, 16)

──────────────────────────────────────
Block 2 第二、三层：1×1 卷积 × 2
──────────────────────────────────────

(1, 256, 16, 16)
    ↓ Conv2d(256→256, kernel=1) + ReLU
(1, 256, 16, 16)
    ↓ Conv2d(256→256, kernel=1) + ReLU
(1, 256, 16, 16)

══════════════════════════════════════
MaxPool（16→8）
══════════════════════════════════════

(1, 256, 16, 16)
    ↓ MaxPool2d(kernel=3, stride=2, padding=1)
(1, 256, 8, 8)

══════════════════════════════════════
Block 3 第一层：3×3 卷积，输出384通道
══════════════════════════════════════

(1, 256, 8, 8)
    ↓ Conv2d(256→384, kernel=3, padding=1)
(1, 384, 8, 8)
    ↓ ReLU
(1, 384, 8, 8)

──────────────────────────────────────
Block 3 第二、三层：1×1 卷积 × 2
──────────────────────────────────────

(1, 384, 8, 8)
    ↓ Conv2d(384→384, kernel=1) + ReLU
(1, 384, 8, 8)
    ↓ Conv2d(384→384, kernel=1) + ReLU
(1, 384, 8, 8)

══════════════════════════════════════
MaxPool（8→4）
══════════════════════════════════════

(1, 384, 8, 8)
    ↓ MaxPool2d(kernel=3, stride=2, padding=1)
(1, 384, 4, 4)

══════════════════════════════════════
Dropout
══════════════════════════════════════

(1, 384, 4, 4)
    ↓ Dropout(p=0.5)
(1, 384, 4, 4)    ← 尺寸不变，只是随机把一些值置零

══════════════════════════════════════
Block 4：这里是关键，输出通道数 = 类别数 = 10
══════════════════════════════════════

(1, 384, 4, 4)
    ↓ Conv2d(384→10, kernel=3, padding=1)
(1, 10, 4, 4)
    ↓ ReLU
(1, 10, 4, 4)

↑ 注意：通道数从 384 变成了 10
  空间尺寸还是 4×4，还没有消掉

    ↓ Conv2d(10→10, kernel=1) + ReLU
(1, 10, 4, 4)
    ↓ Conv2d(10→10, kernel=1) + ReLU
(1, 10, 4, 4)

此时这 10 个通道，每个通道是一张 4×4 的特征图
对应 10 个类别，每张特征图表示"这个类别在各位置的响应强度"

══════════════════════════════════════
全局平均池化（4×4 → 1×1）
══════════════════════════════════════

(1, 10, 4, 4)
    ↓ AdaptiveAvgPool2d((1, 1))

对每个通道的 4×4 = 16 个数字取平均，得到 1 个数字：

  通道0（"飞机"类）：16个数字取平均 → 0.23
  通道1（"汽车"类）：16个数字取平均 → 0.71   ← 最大，预测是汽车
  通道2（"鸟"类） ：16个数字取平均 → 0.05
  ...
  通道9（"卡车"类）：16个数字取平均 → 0.11

(1, 10, 1, 1)

══════════════════════════════════════
Flatten
══════════════════════════════════════

(1, 10, 1, 1)
    ↓ Flatten()
(1, 10)

↑ 这就是最终的 logits，直接喂给 CrossEntropyLoss
  不需要任何全连接层
```

# 04 代码示例

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


# ══════════════════════════════════════════════════════════════════
# 核心模块：NiN Block（MLPConv）
# ══════════════════════════════════════════════════════════════════

def nin_block(
    in_channels : int,
    out_channels: int,
    kernel_size : int,
    stride      : int,
    padding     : int
) -> nn.Sequential:
    """
    构建一个 NiN Block（即 MLPConv 模块）。
    
    结构：Conv(k×k) → ReLU → Conv(1×1) → ReLU → Conv(1×1) → ReLU
    
    前面的 k×k 卷积负责提取局部空间特征（线性变换）。
    后面的两个 1×1 卷积在通道维度上做非线性变换，
    等价于在每个空间位置上运行一个两层 MLP。
    
    Args:
        in_channels : 输入通道数
        out_channels: 输出通道数（三个卷积层的输出通道数相同）
        kernel_size : 第一个卷积的核大小（1×1 卷积的核大小固定为 1）
        stride      : 第一个卷积的步长
        padding     : 第一个卷积的填充
    
    Returns:
        nn.Sequential: 包含三个卷积层和三个 ReLU 的顺序模块
    
    注意：
        三个卷积层的输出通道数都是 out_channels。
        1×1 卷积的 in_channels = out_channels（因为接在第一个卷积的输出后面）。
    """
    return nn.Sequential(
        # ── 第一层：标准局部卷积（提取空间特征）──────────────────
        # kernel_size 可以是 11、5、3 等，取决于 Block 的位置
        nn.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding
        ),
        nn.ReLU(inplace=True),
        # inplace=True：直接在原 tensor 上做 ReLU，节省显存
        # 代价是无法在 autograd 中保留原始激活值，但 ReLU 后一般不需要

        # ── 第二层：第一个 1×1 卷积（通道间非线性组合，第一层MLP）
        nn.Conv2d(
            in_channels  = out_channels,  # 注意：输入通道数 = 上一层的输出通道数
            out_channels = out_channels,  # 通道数保持不变（可以改，但原论文不变）
            kernel_size  = 1              # 1×1 卷积，不改变空间尺寸
        ),
        nn.ReLU(inplace=True),

        # ── 第三层：第二个 1×1 卷积（通道间非线性组合，第二层MLP）
        nn.Conv2d(
            in_channels  = out_channels,
            out_channels = out_channels,
            kernel_size  = 1
        ),
        nn.ReLU(inplace=True),
    )


# ══════════════════════════════════════════════════════════════════
# 完整 NiN 网络
# ══════════════════════════════════════════════════════════════════

class NiN(nn.Module):
    """
    Network in Network 完整实现（适配 CIFAR-10 的修改版）。
    
    原论文针对 224×224 输入设计，这里针对 CIFAR-10 的 32×32 输入做了调整：
        - 第一个卷积核从 11×11 改为 5×5
        - stride 从 4 改为 1
        - 去掉一个 MaxPool
    具体调整原因见代码注释。
    
    网络结构：
        Block1 → MaxPool → Block2 → MaxPool → Block3 → MaxPool
        → Dropout → Block4（输出通道 = 类别数）→ GlobalAvgPool → 输出
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        self.net = nn.Sequential(
            # ── Block 1 ─────────────────────────────────────────────
            # 原论文：kernel=11, stride=4（针对 224×224 输入的大感受野）
            # CIFAR-10 调整：kernel=5, stride=1（输入只有 32×32，不能步长太大）
            # padding=2：保证 5×5 卷积后尺寸不变（32-5+2*2)/1+1=32，正确）
            nin_block(in_channels=3, out_channels=96,
                      kernel_size=5, stride=1, padding=2),
            
            # MaxPool：尺寸减半，32→16
            # kernel_size=3, stride=2：比 2×2 的池化感受野更大，原论文选择
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # padding=1：防止奇数尺寸时丢掉边缘信息，保证 (32+2*1-3)/2+1=16，精确减半

            # ── Block 2 ─────────────────────────────────────────────
            # kernel=5（原论文），捕捉中等大小的空间特征
            # padding=2：(16-5+2*2)/1+1=16，尺寸不变
            nin_block(in_channels=96, out_channels=256,
                      kernel_size=5, stride=1, padding=2),
            
            # MaxPool：16→8
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # ── Block 3 ─────────────────────────────────────────────
            # kernel=3，小卷积核，捕捉细粒度特征
            # padding=1：(8-3+2*1)/1+1=8，尺寸不变
            nin_block(in_channels=256, out_channels=384,
                      kernel_size=3, stride=1, padding=1),
            
            # MaxPool：8→4
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # ── Dropout ──────────────────────────────────────────────
            # NiN 原论文在最后一个 Block 前加 Dropout（p=0.5）
            # 作用：防止过拟合，相当于 VGG 全连接层前的 Dropout
            # NiN 虽然没有全连接层，但 Block4 的参数也不少，需要正则化
            nn.Dropout(p=0.5),

            # ── Block 4（输出层）────────────────────────────────────
            # 关键：out_channels = num_classes（通道数 = 类别数）
            # 这是 NiN 设计的精髓：让最后一个 Block 的每个通道对应一个类别
            # 之后全局平均池化，每个通道的均值就是该类别的置信度
            nin_block(in_channels=384, out_channels=num_classes,
                      kernel_size=3, stride=1, padding=1),

            # ── 全局平均池化 ─────────────────────────────────────────
            # AdaptiveAvgPool2d((1, 1))：输出固定为 1×1，不管输入的 H、W 是多少
            # 对比 AvgPool2d(kernel_size=4)：只能接受固定尺寸 4×4 的输入
            # 用 Adaptive 版本更灵活，是标准写法
            nn.AdaptiveAvgPool2d((1, 1)),
            # 输出：(B, num_classes, 1, 1)

            # ── 展平 ────────────────────────────────────────────────
            # 把 (B, num_classes, 1, 1) 变成 (B, num_classes)
            # 注意：nn.Flatten() 默认从第 1 维开始展平（保留 batch 维）
            nn.Flatten(),
            # 输出：(B, num_classes)，直接作为 logits 输入 CrossEntropyLoss
        )
        
        # ── 权重初始化 ───────────────────────────────────────────────
        # 调用自定义的初始化函数，比默认初始化收敛更快
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        使用 Kaiming 正态分布初始化卷积层权重。
        
        为什么用 Kaiming 而不是默认初始化？
            默认初始化（均匀分布）在深网络中容易导致激活值在前向传播中
            指数级增大或缩小，梯度消失/爆炸问题严重。
            Kaiming 初始化根据输入通道数自动调整方差，
            保证每层输出的方差与输入方差大致相同。
        
        mode='fan_in'：根据输入连接数来设定方差（前向传播更稳定）
        nonlinearity='relu'：专门为 ReLU 激活函数设计
            由于 ReLU 会把一半的值置零，Kaiming 初始化会把方差乘以 2
            来补偿这个"截断效应"
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_in',
                    nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # bias 初始化为 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        输入：(B, 3, H, W)
        输出：(B, num_classes)，即各类别的 logits（未经 Softmax）
        """
        return self.net(x)


# ══════════════════════════════════════════════════════════════════
# 验证网络结构
# ══════════════════════════════════════════════════════════════════

def verify_architecture():
    """打印每一层的输出尺寸，验证网络结构是否正确"""
    model = NiN(num_classes=10)
    
    # 逐层追踪尺寸变化
    x = torch.randn(1, 3, 32, 32)
    print("NiN 前向传播尺寸追踪（输入：1×3×32×32）")
    print("-" * 50)
    
    layer_names = [
        "Block1（5×5卷积+两个1×1）",
        "MaxPool 32→16",
        "Block2（5×5卷积+两个1×1）",
        "MaxPool 16→8",
        "Block3（3×3卷积+两个1×1）",
        "MaxPool 8→4",
        "Dropout",
        "Block4（3×3卷积+两个1×1）",
        "GlobalAvgPool",
        "Flatten",
    ]
    
    for i, (layer, name) in enumerate(zip(model.net, layer_names)):
        x = layer(x)
        print(f"[{i+1:2d}] {name:<30} → {tuple(x.shape)}")
    
    print("-" * 50)
    total = sum(p.numel() for p in model.parameters())
    print(f"总参数量：{total:,}")


verify_architecture()
```

![image-20260304115715917](F:\note\deep_learning\pytorch_learning\day10_NiN.assets\image-20260304115715917.png)

```python
# ══════════════════════════════════════════════════════════════════
# 数据准备
# ══════════════════════════════════════════════════════════════════

class Config:
    DATA_ROOT    = './cifar10_data'
    NUM_CLASSES  = 10
    BATCH_SIZE   = 128
    NUM_EPOCHS   = 60
    LR           = 0.1           # NiN 从头训练，用较大初始学习率
    WEIGHT_DECAY = 5e-4
    MOMENTUM     = 0.9
    DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SAVE_PATH    = './best_nin_cifar10.pth'


CONFIG = Config()


def build_dataloaders():
    """
    构建 CIFAR-10 的训练和测试 DataLoader。

    归一化参数说明：
        从零训练时，应使用数据集本身的统计值，而不是 ImageNet 的值。
        CIFAR-10 的均值和标准差是在 CIFAR-10 全数据集上统计的。
        如果用了预训练权重（迁移学习），才需要用 ImageNet 的归一化参数。
    """
    # CIFAR-10 数据集的 RGB 三通道均值和标准差
    CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
    CIFAR10_STD  = [0.2023, 0.1994, 0.2010]

    train_transform = transforms.Compose([
        # RandomCrop：先在四周各 padding 4 像素，再随机裁回 32×32
        # 这是 CIFAR 上最经典的数据增强，简单有效
        # 作用：让模型学会识别轻微平移后的图像
        transforms.RandomCrop(32, padding=4),

        # 随机水平翻转，概率默认 0.5
        transforms.RandomHorizontalFlip(),

        # 转为 tensor，像素值从 [0,255] 归一化到 [0.0, 1.0]
        transforms.ToTensor(),

        # 按 CIFAR-10 统计量做标准化：(x - mean) / std
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])

    test_transform = transforms.Compose([
        # 测试集不做随机变换，保证每次评估结果一致
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])

    train_ds = CIFAR10(root=CONFIG.DATA_ROOT, train=True,
                       transform=train_transform, download=True)
    test_ds  = CIFAR10(root=CONFIG.DATA_ROOT, train=False,
                       transform=test_transform, download=True)

    train_loader = DataLoader(
        train_ds,
        batch_size  = CONFIG.BATCH_SIZE,
        shuffle     = True,   # 训练集每个 epoch 打乱顺序
        num_workers = 2,
        pin_memory  = (CONFIG.DEVICE.type == 'cuda')
        # pin_memory=True：把数据预加载到固定内存，GPU 传输更快
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = CONFIG.BATCH_SIZE,
        shuffle     = False,  # 测试集不打乱，保证评估可复现
        num_workers = 2,
        pin_memory  = (CONFIG.DEVICE.type == 'cuda')
    )

    print(f"训练集：{len(train_ds)} 张 | 测试集：{len(test_ds)} 张")
    return train_loader, test_loader


# ══════════════════════════════════════════════════════════════════
# 训练一个 epoch
# ══════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer):
    """
    训练一个完整的 epoch，返回（平均loss，准确率%）。
    """
    model.train()
    # model.train() 的作用：
    #   开启 Dropout（随机置零神经元）
    #   BatchNorm 使用当前 batch 的统计量（如果有 BN 层）
    # 每次进入训练循环前必须调用，否则 Dropout 不生效

    total_loss, correct, total = 0.0, 0, 0

    for inputs, labels in loader:
        inputs = inputs.to(CONFIG.DEVICE)
        labels = labels.to(CONFIG.DEVICE)

        # 梯度清零：必须在每个 batch 的前向传播前调用
        # PyTorch 默认累积梯度，不清零会导致梯度叠加，训练出错
        # set_to_none=True：比 zero_grad() 更高效，直接释放梯度内存
        optimizer.zero_grad(set_to_none=True)

        outputs = model(inputs)              # 前向传播，输出 (B, 10)
        loss    = criterion(outputs, labels) # 计算交叉熵损失

        loss.backward()  # 反向传播，计算所有参数的梯度

        # 梯度裁剪：防止梯度爆炸
        # NiN 使用较大初始 lr=0.1，训练初期梯度可能很大
        # max_norm=5.0：将所有参数梯度的全局 L2 范数裁剪到不超过 5.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()  # 更新参数：param = param - lr × grad

        total_loss += loss.item() * inputs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total * 100


# ══════════════════════════════════════════════════════════════════
# 验证 / 测试
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, criterion):
    """
    在给定数据集上评估模型，返回（平均loss，准确率%）。

    @torch.no_grad()：
        禁用梯度计算，不存储中间激活值
        验证时不需要反向传播，这样做节省显存、加快速度
    """
    model.eval()
    # model.eval() 的作用：
    #   关闭 Dropout（所有神经元都参与计算，结果稳定可复现）
    #   BatchNorm 使用训练阶段统计的全局均值和方差（而非当前 batch）
    # 易错点：如果忘记调用 model.eval()，
    #   Dropout 还在随机 drop 神经元，每次前向传播结果不同，评估不准确

    total_loss, correct, total = 0.0, 0, 0

    for inputs, labels in loader:
        inputs  = inputs.to(CONFIG.DEVICE)
        labels  = labels.to(CONFIG.DEVICE)
        outputs = model(inputs)
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * inputs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total * 100


# ══════════════════════════════════════════════════════════════════
# 主训练流程
# ══════════════════════════════════════════════════════════════════

def main():
    print(f"使用设备：{CONFIG.DEVICE}\n")

    # 验证网络结构
    verify_architecture()

    # 加载数据
    train_loader, test_loader = build_dataloaders()

    # 构建模型
    model = NiN(num_classes=CONFIG.NUM_CLASSES).to(CONFIG.DEVICE)

    # 损失函数
    # CrossEntropyLoss 内部已包含 Softmax + NLLLoss
    # 所以模型输出不要加 Softmax，直接输出 logits 即可
    criterion = nn.CrossEntropyLoss()

    # 优化器：SGD，NiN 从头训练用大学习率 0.1
    # 如果用 Adam(lr=1e-4)，收敛会非常慢（参数量少，SGD 更合适）
    optimizer = optim.SGD(
        model.parameters(),
        lr           = CONFIG.LR,
        momentum     = CONFIG.MOMENTUM,
        weight_decay = CONFIG.WEIGHT_DECAY
    )

    # 学习率调度：余弦退火
    # 学习率从 0.1 按余弦曲线平滑下降到 1e-5
    # 比 StepLR（阶梯下降）更平滑，后期收敛更精细
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG.NUM_EPOCHS, eta_min=1e-5)

    best_acc = 0.0

    for epoch in range(1, CONFIG.NUM_EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        te_loss, te_acc = evaluate(model, test_loader, criterion)

        # scheduler.step() 每个 epoch 结束后调用一次（不是每个 batch）
        scheduler.step()

        print(f"Epoch {epoch:3d}/{CONFIG.NUM_EPOCHS} | "
              f"Train Loss {tr_loss:.4f} Acc {tr_acc:.2f}% | "
              f"Test  Loss {te_loss:.4f} Acc {te_acc:.2f}%")

        if te_acc > best_acc:
            best_acc = te_acc
            # 只保存模型参数（state_dict），不保存整个模型对象
            # 这样加载时不依赖代码版本，更灵活
            torch.save(model.state_dict(), CONFIG.SAVE_PATH)
            print(f"  ✓ 保存最优模型（Test Acc: {best_acc:.2f}%）")

    print(f"\n训练完成，最优测试准确率：{best_acc:.2f}%")
    # CIFAR-10 上 NiN 正常可以达到约 89%~91%


if __name__ == '__main__':
    main()
```

## 04.1 nn.AdaptiveAvgPool2d

```python
# AvgPool2d：需要你指定窗口大小和步长，输出尺寸依赖输入尺寸
pool_fixed = nn.AvgPool2d(kernel_size=4, stride=4)
# 输入 (B, C, 4, 4) → 输出 (B, C, 1, 1)  ✓
# 输入 (B, C, 8, 8) → 输出 (B, C, 2, 2)  ✗（不是我们想要的）
# 输入 (B, C, 7, 7) → 报错或结果不对      ✗

# AdaptiveAvgPool2d：你指定输出尺寸，PyTorch 自动计算窗口大小
pool_adaptive = nn.AdaptiveAvgPool2d((1, 1))
# 输入 (B, C, 4, 4)  → 输出 (B, C, 1, 1)  ✓
# 输入 (B, C, 8, 8)  → 输出 (B, C, 1, 1)  ✓（自动用 8×8 的窗口）
# 输入 (B, C, 7, 7)  → 输出 (B, C, 1, 1)  ✓（自动用 7×7 的窗口）
# 这就是为什么现代网络几乎都用 Adaptive 版本
```

## 04.2 nn.Flatten

```python
# nn.Flatten(start_dim=1, end_dim=-1)
# start_dim=1：从第 1 维开始展平（保留 batch 维度，即第 0 维）
# end_dim=-1 ：一直展平到最后一维

flatten = nn.Flatten()   # 等价于 nn.Flatten(start_dim=1)

x = torch.randn(4, 10, 1, 1)   # batch=4, channels=10, 1×1 空间
out = flatten(x)
print(out.shape)  # torch.Size([4, 10])  ← batch 维保留，其余展平

# 易错点：如果写 nn.Flatten(start_dim=0) 会把 batch 维也展平，
# 导致整个 batch 变成一个长向量，训练时会报维度错误
```







