# 01 GoogLeNet背景

## 01.1 问题一：卷积核大小的选择

在 VGG 里，统一用 3×3 卷积，靠堆叠层数来增大感受野。这个做法简单有效，但有一个隐含的假设：==所有局部特征都应该用同样大小的感受野来提取==

现实中这个假设不成立。一张图里，有些目标很小（比如远处的一只鸟），有些目标很大（比如近处的一辆车）。用同一个卷积核大小去处理，必然有一个尺度是不合适的。

**GoogLeNet 的出发点就是：同一层，能不能同时用多种大小的卷积核，让网络自己去学哪个尺度更重要？**

## 01.2 问题二：并行多种卷积计算量爆炸

最直觉的想法：在同一层里并排放 1×1、3×3、5×5 三个卷积，把输出拼起来。

但这样做有一个严重问题：**计算量爆炸**。

```css
假设输入特征图：(B, 192, 28, 28)

直接并行三种卷积（各输出 128 通道）：
  1×1 卷积：1×1×192×128          =   3,072,000  次乘加
  3×3 卷积：3×3×192×128          =  27,648,000  次乘加
  5×5 卷积：5×5×192×128          =  76,800,000  次乘加
  合计：约 1.07 亿次乘加（仅这一层）
```

堆叠这样的层，整个网络的计算量会大得无法接受。

**GoogLeNet 的解法：在每个大卷积之前，先用 1×1 卷积把通道数压下来，计算完再恢复。** 这就是 Inception 模块的核心设计

# 02 inception模块

Inception 模块有四条并行的路径，最后把四条路径的输出在**通道维度**拼接在一起：

```css
输入特征图 (B, C_in, H, W)
         │
    ┌────┴─────────────────────────────┐
    │         │           │            │
    ▼         ▼           ▼            ▼
  1×1卷积   1×1卷积     1×1卷积    3×3最大池化
  (降维)    (降维)      (降维)
    │         │           │            │
    │       3×3卷积     5×5卷积      1×1卷积
    │                                (补充池化后的通道融合)   
    │         │           │            │
    └────┬─────────────────────────────┘
         │
    通道维度拼接（Concat）
         │
    输出特征图 (B, C_out, H, W)
    C_out = 四条路径输出通道数之和
```

四条路径的作用：

- **路径一（1×1 卷积）**：提取点级特征，感受野最小，计算量最小
- **路径二（1×1 → 3×3）**：提取中等感受野特征，1×1 先降维减少计算量
- **路径三（1×1 → 5×5）**：提取大感受野特征，1×1 先降维减少计算量
- **路径四（3×3 MaxPool → 1×1）**：通过池化保留位置不变性特征，1×1 调整通道数

## 02.1 通道拼接（Concat）

```css
输入：(B, 192, 28, 28)

路径一：1×1 卷积，输出 64 通道
  Conv(192→64, kernel=1, padding=0)
  输出：(B, 64, 28, 28)
  H、W 不变：(28 - 1 + 0) / 1 + 1 = 28 ✓

路径二：1×1 降维到 96，再 3×3 卷积输出 128
  Conv(192→96, kernel=1)  → (B, 96,  28, 28)
  Conv(96→128, kernel=3, padding=1) → (B, 128, 28, 28)
  H、W 不变：(28 - 3 + 2) / 1 + 1 = 28 ✓

路径三：1×1 降维到 16，再 5×5 卷积输出 32
  Conv(192→16, kernel=1)  → (B, 16, 28, 28)
  Conv(16→32,  kernel=5, padding=2) → (B, 32,  28, 28)
  H、W 不变：(28 - 5 + 4) / 1 + 1 = 28 ✓

路径四：3×3 MaxPool，再 1×1 卷积输出 32
  MaxPool(kernel=3, stride=1, padding=1) → (B, 192, 28, 28)
  H、W 不变：(28 - 3 + 2) / 1 + 1 = 28 ✓
  Conv(192→32, kernel=1) → (B, 32, 28, 28)

最终四条路径的输出：
  路径一：(B,  64, 28, 28)
  路径二：(B, 128, 28, 28)
  路径三：(B,  32, 28, 28)
  路径四：(B,  32, 28, 28)

通道数各不相同，但 B、H、W 全部相同（都是 28×28）
→ 可以在通道维度拼接！

拼接后：(B, 64+128+32+32, 28, 28) = (B, 256, 28, 28)
```

**H 和 W 对不对得上，是靠每条路径里的 padding 精确控制的，不是自动对好的。这是写代码时必须手动保证的。** 如果任何一条路径的 padding 写错，H 或 W 就会不一样，拼接时就会报错



**记忆规律：padding = kernel_size // 2，可以保证空间尺寸不变**

- 1×1 卷积：padding = 0
- 3×3 卷积：padding = 1
- 5×5 卷积：padding = 2



`torch.cat`：

是 PyTorch 中最常用且最重要的张量操作之一，全称是 **concatenate（拼接/级联）**。它的核心作用是将多个张量（Tensor）沿着**已经存在**的某一个维度拼接在一起



在 CNN 中，图像或特征图的张量维度通常是 **`[B, C, H, W]`**：

- `B` (Batch Size): 批次大小
- `C` (Channels): 通道数（特征图数量）
- `H` (Height): 高度
- `W` (Width): 宽度

在网络结构中，我们最常做的是**按通道维度拼接（`dim=1`）**，这意味着把不同卷积层提取到的不同特征“叠”在一起

```python
# 模拟深度学习中的特征图
# 假设 batch_size=4, 图片大小 32x32
# 特征图 A 提取了 16 个通道的特征
feat_A = torch.randn(4, 16, 32, 32) 

# 特征图 B 提取了 32 个通道的特征
# 注意：B、H、W 必须与 feat_A 一致！
feat_B = torch.randn(4, 32, 32, 32) 

# 沿着通道维度 (dim=1) 拼接
fused_feat = torch.cat((feat_A, feat_B), dim=1)

print(fused_feat.shape) 
# 输出: torch.Size([4, 48, 32, 32]) (16通道 + 32通道 = 48通道)
```

## 02.2 1×1卷积 降维减少计算量

```css
不压缩，直接做 5×5 卷积（输入 192 通道，输出 32 通道）：
  每个输出值 = 5×5×192 个数字的加权求和
  计算量 = 5×5×192×32×28×28 = 120,422,400 次

先用 1×1 卷积压缩到 16 通道，再做 5×5 卷积（输出 32 通道）：
  第一步 1×1 卷积：1×1×192×16×28×28 =   2,408,448 次
  第二步 5×5 卷积：5×5×16×32×28×28  =  10,035,200 次
  合计：                                  12,443,648 次

节省了约 90% 的计算量
```

# 03 辅助分类器

GoogLeNet 有 22 层。训练时，梯度从最后一层开始，一层一层往回传（反向传播）。每经过一层，梯度值都会乘以一个小于 1 的数（因为激活函数的导数通常小于 1），走到第 5、6 层时，梯度可能已经小到接近于零



辅助分类器的思路非常直接：**在网络的中间位置，额外接一个小型分类器，让它也计算损失、也产生梯度。** 这样浅层参数除了从最终输出接收梯度之外，还能从中间的辅助分类器接收梯度，梯度传播路径更短，不容易消失

```css
普通网络的梯度传播路径（只有一条）：

第1层 ← 第2层 ← ... ← 第15层 ← 第20层 ← 输出损失
  ↑                                              |
  └──────────────── 梯度要走很远 ─────────────────┘
        （走了 20 层，梯度已经很小了）

GoogLeNet 的梯度传播路径（有三条）：

第1层 ← 第2层 ← ... ← 第10层 ← 第20层 ← 主输出损失
                          ↑
                    辅助分类器1损失  ← 第10层只需走10步就能收到梯度
                          
第1层 ← ... ← 第15层 ← 第20层 ← 主输出损失
                  ↑
            辅助分类器2损失  ← 第15层只需走5步就能收到梯度
```



训练时：

```css
训练时计算三个损失：

主分类器损失：CrossEntropy(主输出, 真实标签)
辅助损失1   ：CrossEntropy(辅助输出1, 真实标签)
辅助损失2   ：CrossEntropy(辅助输出2, 真实标签)

总损失 = 主损失 + 0.3 × 辅助损失1 + 0.3 × 辅助损失2

为什么辅助损失的权重是 0.3 而不是 1.0？
  如果三个损失权重相同，辅助分类器会对训练产生过大干扰。
  0.3 的权重让辅助分类器只是"帮助梯度传播"，
  而不是"主导训练方向"。
```



推理时：

推理时只用主分类器的输出。辅助分类器的两个分支完全跳过，就好像不存在一样。

这和 Dropout 的逻辑类似：训练和推理时网络行为不同。PyTorch 里用 `self.training` 这个属性来控制



假设网络只有 5 层，辅助分类器接在第 3 层后面：

```python
# 简化版，帮助理解辅助分类器的原理
import torch
import torch.nn as nn

class SimpleNetWithAux(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 10)
        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, 10)   # ← 辅助分类器接在这里
        self.layer4 = nn.Linear(10, 10)
        self.layer5 = nn.Linear(10, 3)    # 主输出，3个类别

        # 辅助分类器：一个小型分类头，接在 layer3 后面
        self.aux_classifier = nn.Linear(10, 3)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        # ↑ 到这里，x 是 layer3 的输出

        # 辅助分类器在这里分叉：
        # x 同时被送进辅助分类器 AND 继续往后走
        aux_out = self.aux_classifier(x)
        # aux_out 是辅助分类器的输出（分类结果）
        # 注意：x 本身没有被修改，只是被"读了一下"

        x = torch.relu(self.layer4(x))
        x = self.layer5(x)               # 主输出

        return x, aux_out
```



前向传播的数据流动是这样的：

```css
输入
 ↓
layer1
 ↓
layer2
 ↓
layer3 ──── aux_classifier ──→ aux_out（辅助输出）
 ↓
layer4
 ↓
layer5 ──────────────────────→ main_out（主输出）
```



训练时的损失计算：

```python
model = SimpleNetWithAux()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

inputs = torch.randn(4, 10)    # 4个样本
labels = torch.tensor([0, 1, 2, 0])

# 前向传播，得到两个输出
main_out, aux_out = model(inputs)

# 分别计算两个损失
main_loss = criterion(main_out, labels)
aux_loss  = criterion(aux_out,  labels)

# 合并成总损失
# 0.3 是辅助损失的权重，让它只起辅助作用，不主导训练
total_loss = main_loss + 0.3 * aux_loss

# 反向传播
total_loss.backward()
```



layer1 和 layer2 收到的梯度是两条路径梯度的**叠加**：

```css
layer1 的梯度 = （来自 main_loss 的梯度，走了5层，已经很小）
              + （来自 aux_loss 的梯度，只走了3层，相对较大）

= 一个很小的数 + 一个相对大的数
= 比没有辅助分类器时大得多
```

> * 我理解这个辅助分类器就是在浅层直接学习输入对输出（这里要把输出设为最终分类数），因为总的输出在反向传播时，到达浅层的数值可能非常小，无法进一步优化，这时候就可以把辅助分类器的一并纳入进去，帮助浅层的参数更好优化

# 04 完整代码

```css
GoogLeNet 的整体结构：

输入图片
  ↓
Stem（几个普通卷积，快速降低空间尺寸）
  ↓
Inception模块 → Inception模块
  ↓ MaxPool（空间减半）
Inception模块 → Inception模块 → Inception模块 → Inception模块 → Inception模块
  ↓ MaxPool
Inception模块 → Inception模块
  ↓
全局平均池化
  ↓
全连接层（只有一个）
  ↓
输出（分类结果）
```



```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


# ══════════════════════════════════════════════════════════════════
# Inception 模块
# ══════════════════════════════════════════════════════════════════

class InceptionBlock(nn.Module):
    """
    Inception 模块：四条并行路径，最后在通道维度拼接。

    Args:
        in_channels : 输入通道数
        c1          : 路径1，1×1 卷积输出通道数
        c2_reduce   : 路径2，1×1 降维后的通道数（不出现在最终输出里）
        c2          : 路径2，3×3 卷积最终输出通道数
        c3_reduce   : 路径3，1×1 降维后的通道数（不出现在最终输出里）
        c3          : 路径3，5×5 卷积最终输出通道数
        c4          : 路径4，1×1 卷积输出通道数

    输出通道数 = c1 + c2 + c3 + c4
    注意：c2_reduce 和 c3_reduce 只是降维用的中间值，不计入输出！
    """
    def __init__(self, in_channels, c1, c2_reduce, c2, c3_reduce, c3, c4):
        super().__init__()

        # ── 路径一：1×1 卷积 ─────────────────────────────────────
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=1),
            # kernel=1, padding=0：天然不改变空间尺寸
            nn.ReLU(inplace=True)
        )

        # ── 路径二：1×1 降维 → 3×3 卷积 ─────────────────────────
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels, c2_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2_reduce, c2, kernel_size=3, padding=1),
            # padding=1：(H-3+2)/1+1=H，尺寸不变
            # 必须是 padding=1，是保证尺寸不变的关键
            nn.ReLU(inplace=True)
        )

        # ── 路径三：1×1 降维 → 5×5 卷积 ─────────────────────────
        self.path3 = nn.Sequential(
            nn.Conv2d(in_channels, c3_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3_reduce, c3, kernel_size=5, padding=2),
            # padding=2：(H-5+4)/1+1=H，尺寸不变
            # 必须是 padding=2，不是 1！
            nn.ReLU(inplace=True)
        )

        # ── 路径四：MaxPool → 1×1 卷积 ───────────────────────────
        self.path4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            # stride=1（关键！不是 2）：只聚合信息，不缩小空间尺寸
            # padding=1：保证 MaxPool 后尺寸不变
            nn.Conv2d(in_channels, c4, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        p1 = self.path1(x)   # (B, c1, H, W)
        p2 = self.path2(x)   # (B, c2, H, W)
        p3 = self.path3(x)   # (B, c3, H, W)
        p4 = self.path4(x)   # (B, c4, H, W)

        # 四条路径并行计算（互不干扰），在通道维度拼接
        # dim=1：通道维度（第1维）   N C H W
        # 前提：四条路径的 H 和 W 必须完全相同（靠 padding 保证）
        return torch.cat([p1, p2, p3, p4], dim=1)
        # 输出：(B, c1+c2+c3+c4, H, W)


# ══════════════════════════════════════════════════════════════════
# 辅助分类器
# ══════════════════════════════════════════════════════════════════

class AuxiliaryClassifier(nn.Module):
    """
    辅助分类器，接在 Inception 模块的中间位置。

    作用：在网络中间产生一个损失值，给浅层参数提供一条路径更短、
         梯度衰减更少的梯度传播路径，缓解梯度消失问题。

    只在训练时参与计算（self.training=True 时），
    推理时完全跳过（forward 里用 if self.training 控制）。

    Args:
        in_channels : 接入点的通道数
        num_classes : 分类数
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.net = nn.Sequential(
            # 5×5 AvgPool，stride=3：快速缩小空间，减少后续计算量
            # 输入 (B, in_channels, 8, 8)
            # 输出尺寸：(8-5)/3 + 1 = 2，输出 (B, in_channels, 2, 2)
            nn.AvgPool2d(kernel_size=5, stride=3),

            # 1×1 卷积压缩通道
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.ReLU(inplace=True),

            # 展平：128 × 2 × 2 = 512
            # 易错点：展平后的维度 = 通道数 × 空间H × 空间W
            # 不能只写通道数 128，还要乘以空间维度！
            nn.Flatten(),

            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),
        )
# 完整 GoogLeNet
# ══════════════════════════════════════════════════════════════════

class GoogLeNet(nn.Module):
    """
    GoogLeNet 完整实现（针对 CIFAR-10 的 32×32 输入简化版）。

    两大核心创新：
        1. Inception 模块：四条并行路径，多尺度特征提取，
                          用 1×1 卷积降维控制计算量
        2. 辅助分类器：接在中间层，缓解梯度消失，训练完就扔掉

    Args:
        num_classes : 分类数
        aux_logits  : 是否启用辅助分类器
                      训练时 True，推理时靠 self.training 自动控制
    """
    def __init__(self, num_classes=10, aux_logits=True):
        super().__init__()
        self.aux_logits = aux_logits

        # ── Stem ─────────────────────────────────────────────────
        # 原论文 Stem 针对 224×224，有 7×7 stride=2 大卷积
        # 32×32 不能用大步长，改为小卷积
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 输出：(B, 192, 16, 16)
        )

        # ── Inception 模块组 1 ───────────────────────────────────
        # 参数顺序：(in, c1, c2_reduce, c2, c3_reduce, c3, c4)
        # 输出通道 = c1 + c2 + c3 + c4（不含 c2_reduce 和 c3_reduce！）
        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        # 输出通道：64+128+32+32 = 256，空间 16×16
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        # 输出通道：128+192+96+64 = 480，空间 16×16
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 输出：(B, 480, 8, 8)

        # ── Inception 模块组 2 ───────────────────────────────────
        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        # 输出通道：192+208+48+64 = 512

        # 辅助分类器 1 接在 inception4a 后面
        if aux_logits:
            self.aux1 = AuxiliaryClassifier(512, num_classes)

        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        # 输出通道：160+224+64+64 = 512
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        # 输出通道：128+256+64+64 = 512
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        # 输出通道：112+288+64+64 = 528

        # 辅助分类器 2 接在 inception4d 后面
        if aux_logits:
            self.aux2 = AuxiliaryClassifier(528, num_classes)

        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        # 输出通道：256+320+128+128 = 832
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 输出：(B, 832, 4, 4)

        # ── Inception 模块组 3 ───────────────────────────────────
        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        # 输出通道：256+320+128+128 = 832
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)
        # 输出通道：384+384+128+128 = 1024

        # ── 输出部分 ──────────────────────────────────────────────
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 不管输入 H、W 是多少，都输出 1×1

        self.dropout = nn.Dropout(p=0.4)

        # 只有一个全连接层（VGG 有三个）
        # 靠全局平均池化消掉空间维度，不需要大量 FC
        self.fc = nn.Linear(1024, num_classes)

        self._init_weights()

    def _init_weights(self):
        """
        Xavier 正态分布初始化。
        GoogLeNet 原论文使用 Xavier 初始化。
        现代实践中也可以用 Kaiming（专为 ReLU 设计），两者效果差别不大。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        训练模式（self.training=True）：返回 (主输出, 辅助输出1, 辅助输出2)
        推理模式（self.training=False）：只返回主输出

        self.training 由 PyTorch 自动维护：
            model.train() → self.training = True
            model.eval()  → self.training = False
        不需要手动设置。
        """
        # Stem
        x = self.stem(x)                    # (B, 192, 16, 16)

        # Inception 组 1
        x = self.inception3a(x)             # (B, 256, 16, 16)
        x = self.inception3b(x)             # (B, 480, 16, 16)
        x = self.maxpool1(x)                # (B, 480, 8, 8)

        # Inception 组 2
        x = self.inception4a(x)             # (B, 512, 8, 8)

        aux1_out = None
        if self.aux_logits and self.training:
            # x 被送进辅助分类器，但 x 本身没有被修改
            # 辅助分类器只是在旁边叉了一条支路读了一下 x
            aux1_out = self.aux1(x)

        x = self.inception4b(x)             # (B, 512, 8, 8)
        x = self.inception4c(x)             # (B, 512, 8, 8)
        x = self.inception4d(x)             # (B, 528, 8, 8)

        aux2_out = None
        if self.aux_logits and self.training:
            aux2_out = self.aux2(x)

        x = self.inception4e(x)             # (B, 832, 8, 8)
        x = self.maxpool2(x)                # (B, 832, 4, 4)

        # Inception 组 3
        x = self.inception5a(x)             # (B, 832, 4, 4)
        x = self.inception5b(x)             # (B, 1024, 4, 4)

        # 输出部分
        x = self.avgpool(x)                 # (B, 1024, 1, 1)
        x = torch.flatten(x, 1)            # (B, 1024)
        x = self.dropout(x)
        x = self.fc(x)                     # (B, num_classes)

        if self.aux_logits and self.training:
            return x, aux1_out, aux2_out   # 训练：三个输出
        return x                           # 推理：只有主输出


# ══════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════

class Config:
    DATA_ROOT    = './cifar10_data'
    NUM_CLASSES  = 10
    BATCH_SIZE   = 128
    NUM_EPOCHS   = 60
    LR           = 0.1
    WEIGHT_DECAY = 1e-4
    MOMENTUM     = 0.9
    AUX_WEIGHT   = 0.3     # 辅助损失权重，原论文使用 0.3
    DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SAVE_PATH    = './best_googlenet_cifar10.pth'

CONFIG = Config()


# ══════════════════════════════════════════════════════════════════
# 数据准备
# ══════════════════════════════════════════════════════════════════

def build_dataloaders():
    CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
    CIFAR10_STD  = [0.2023, 0.1994, 0.2010]

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_ds = CIFAR10(CONFIG.DATA_ROOT, train=True,  transform=train_tf, download=True)
    test_ds  = CIFAR10(CONFIG.DATA_ROOT, train=False, transform=test_tf,  download=True)

    train_loader = DataLoader(train_ds, batch_size=CONFIG.BATCH_SIZE,
                              shuffle=True,  num_workers=2,
                              pin_memory=(CONFIG.DEVICE.type == 'cuda'))
    test_loader  = DataLoader(test_ds,  batch_size=CONFIG.BATCH_SIZE,
                              shuffle=False, num_workers=2,
                              pin_memory=(CONFIG.DEVICE.type == 'cuda'))
    return train_loader, test_loader


# ══════════════════════════════════════════════════════════════════
# 训练一个 epoch
# ══════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    # model.train() 之后：
    #   self.training = True
    #   forward 里的辅助分类器分支会被执行
    #   forward 返回 (main_out, aux1_out, aux2_out) 三个值
    #   Dropout 开启

    total_loss, correct, total = 0.0, 0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(CONFIG.DEVICE), labels.to(CONFIG.DEVICE)
        optimizer.zero_grad(set_to_none=True)

        # 训练时必须解包三个返回值
        # 忘记解包会报错：criterion 收到 tuple 而不是 tensor
        main_out, aux1_out, aux2_out = model(inputs)

        main_loss = criterion(main_out,  labels)
        aux1_loss = criterion(aux1_out,  labels)
        aux2_loss = criterion(aux2_out,  labels)

        # 总损失 = 主损失 + 0.3×辅助损失1 + 0.3×辅助损失2
        # backward() 时梯度会沿三条路径同时传播
        loss = main_loss + CONFIG.AUX_WEIGHT * (aux1_loss + aux2_loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # 统计时只看主分类器的 loss 和 acc，方便和验证集对比
        total_loss += main_loss.item() * inputs.size(0)
        correct    += (main_out.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total * 100


# ══════════════════════════════════════════════════════════════════
# 评估
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    # model.eval() 之后：
    #   self.training = False
    #   forward 里辅助分类器分支被跳过
    #   forward 只返回主输出（一个值，不是三个）
    #   Dropout 关闭

    total_loss, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(CONFIG.DEVICE), labels.to(CONFIG.DEVICE)
        outputs = model(inputs)   # eval 模式只返回一个值
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * inputs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total * 100


# ══════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════

def main():
    print(f"使用设备：{CONFIG.DEVICE}")

    # 验证输出尺寸和参数量
    m = GoogLeNet(10, aux_logits=False)
    m.eval()
    out = m(torch.randn(1, 3, 32, 32))
    total = sum(p.numel() for p in m.parameters())
    print(f"输出尺寸：{tuple(out.shape)}")
    print(f"总参数量：{total:,}（约 {total/1e6:.2f}M）")
    print("对比：VGG-16 约 138M，GoogLeNet 约 6M，参数量约为 VGG 的 1/23\n")

    train_loader, test_loader = build_dataloaders()
    model = GoogLeNet(CONFIG.NUM_CLASSES, aux_logits=True).to(CONFIG.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=CONFIG.LR,
                          momentum=CONFIG.MOMENTUM, weight_decay=CONFIG.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG.NUM_EPOCHS, eta_min=1e-5)

    best_acc = 0.0
    for epoch in range(1, CONFIG.NUM_EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        te_loss, te_acc = evaluate(model, test_loader, criterion)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{CONFIG.NUM_EPOCHS} | "
              f"Train {tr_loss:.4f}/{tr_acc:.2f}% | "
              f"Test  {te_loss:.4f}/{te_acc:.2f}%")

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), CONFIG.SAVE_PATH)
            print(f"  ✓ 保存最优模型（Test Acc: {best_acc:.2f}%）")

    print(f"\n训练完成，最优测试准确率：{best_acc:.2f}%")
    # CIFAR-10 上 GoogLeNet 正常可以达到约 93%~95%


if __name__ == '__main__':
    main()
```

# 05 处理梯度消失的方法对比

![image-20260304180753553](F:\note\deep_learning\pytorch_learning\day11_GoogLeNet.assets\image-20260304180753553.png)



