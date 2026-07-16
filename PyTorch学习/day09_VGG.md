# 01 VGG

VGG 由牛津大学视觉几何组（Visual Geometry Group）在 2014 年提出，论文名为《Very Deep Convolutional Networks for Large-Scale Image Recognition》。它在 ImageNet 2014 比赛中取得了亚军，但其**网络结构设计思想影响深远**，至今仍是理解现代 CNN 的基石



VGG 将网络分成若干"块（Block）"，每块由若干个相同的卷积层 + 一个最大池化层组成。这种**模块化设计思想**直接启发了后来的 ResNet、VGG-style backbone 等



VGG 系列从 VGG-11 到 VGG-19，系统地探索了"深度"对性能的影响，证明了**深度是提升性能的关键因素**之一

## 01.1 感受野

> * 感受野是指：输出特征图上的**某一个像素点**，它的值是由输入图像上**哪个区域**的像素共同决定的。这个区域的大小就是感受野大小
> * 直觉上理解：感受野越大，意味着这个神经元"看到"的原始图像范围越广，能捕捉到更全局的语义信息
> * 三个 3×3卷积之间有**三个 ReLU**，而一个 7×7卷积只有**一个 ReLU**。更多的非线性意味着模型可以拟合更复杂的函数边界

![image-20260226180736096](F:\note\deep_learning\pytorch_learning\day09_VGG.assets\image-20260226180736096.png)

## 01.2 VGG16网络结构

`Conv3-64` 的意思是：使用 **3×3 的卷积核**，输出通道数为 **64**
 `× 2` 表示这样的卷积层**连续叠加 2 次**
 `→ MaxPool` 表示卷积块结束后接一个最大池化层（kernel=2, stride=2，尺寸减半）

```makefile
输入: (224, 224, 3)

Block1: Conv3-64 × 2  → MaxPool  输出: (112, 112, 64)
Block2: Conv3-128 × 2 → MaxPool  输出: (56,  56,  128)
Block3: Conv3-256 × 3 → MaxPool  输出: (28,  28,  256)
Block4: Conv3-512 × 3 → MaxPool  输出: (14,  14,  512)
Block5: Conv3-512 × 3 → MaxPool  输出: (7,   7,   512)

Flatten → FC(4096) → FC(4096) → FC(1000) → Softmax

FC(4096)  ：25088 → 4096，接 ReLU + Dropout
FC(4096)  ：4096  → 4096，接 ReLU + Dropout
FC(1000)  ：4096  → 1000（ImageNet 的 1000 个类别）
Softmax   ：输出每个类别的概率，所有概率之和 = 1
```

![image-20260226192452737](F:\note\deep_learning\pytorch_learning\day09_VGG.assets\image-20260226192452737.png)

**规律一：** 卷积层（padding=1）**不改变** H 和 W，只改变通道数 C。
 **规律二：** MaxPool（stride=2）**让 H 和 W 各减半**，不改变通道数 C。

所以整张图的空间尺寸变化完全由 5 次 MaxPool 决定：

224→112→56→28→14→7

而通道数的变化规律是：

3→64→128→256→512→512 

每个 Block 通道数翻倍（到 512 后不再翻），这样设计是因为**空间尺寸缩小的同时，用更多通道来补偿信息容量**，保证网络的表达能力不随池化而退化。

## 01.3 nn.Sequential

`nn.Sequential` 就是一个“**按顺序串起来执行**”的容器：把若干层/模块按顺序塞进去，它的 `forward` 就是依次把输入丢给下一个模块



适合场景：

- 网络结构是**线性的链式结构**（典型 CNN、MLP 的一段）
- 你不想手写 `forward`（或者 `forward` 很简单）

不适合场景：

- 有 **多分支/跳连**（ResNet 的 skip、UNet 的 concat、注意力分支等）
- 需要在 forward 里做动态控制（if/for、根据输入改变路径）

### 1. 模板1

> * MLP（最简单的 Sequential）

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim=784, num_classes=10):
        super().__init__()
        # 一个典型全连接分类器：in -> 512 -> 256 -> classes
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: [N, in_dim]
        return self.net(x)

# quick test
model = MLP()
x = torch.randn(8, 784)
logits = model(x)
print(logits.shape)  # [8, 10]
```

### 2. 模板2

> * CNN Block（Conv-BN-ReLU）重复堆叠

```python
import torch.nn as nn

def conv_bn_relu(in_ch, out_ch, k=3, s=1, p=1):
    """
    返回一个典型的 Conv2d + BN + ReLU block
    - in_ch/out_ch: 输入/输出通道
    - k/s/p: 卷积核/步幅/padding
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

block = conv_bn_relu(3, 64)
```

### 3. 模板3

> * VGG 风格 `make_layers(cfg)`

```python
import torch.nn as nn

def make_vgg_features(cfg, in_channels=3, use_bn=False):
    """
    根据 cfg 列表构造 VGG 的 features
    cfg 例子: [64, 64, "M", 128, 128, "M", ...]
    - 数字: 输出通道数，代表一个 conv3x3
    - "M": MaxPool2d
    """
    layers = []
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if use_bn:
                layers += [conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv, nn.ReLU(inplace=True)]
            in_channels = v

    # 最终把 list 变成 Sequential
    return nn.Sequential(*layers)
```

### 4. 模板4

> * 迁移学习中替换 VGG classifier（Sequential 的“改头”操作）

```python
import torch.nn as nn
from torchvision import models

weights = models.VGG16_Weights.DEFAULT
model = models.vgg16(weights=weights)

num_classes = 200  # 假设 Tiny-ImageNet 200 类

# VGG classifier 本身就是 nn.Sequential
print(model.classifier)

# 替换最后一层 Linear(4096 -> 1000) 为 Linear(4096 -> num_classes)
in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, num_classes)

# 可选：如果你想“缩小 head”防过拟合，也可以整个替换
model.classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 1024),
    nn.ReLU(True),
    nn.Dropout(0.5),
    nn.Linear(1024, num_classes),
)
```

## 01.4 make_layers 动态构建

如果要手写 VGG-11、VGG-13、VGG-16、VGG-19 四个模型，最笨的做法是复制粘贴四份代码，每份里面手动写出所有层。这样做的问题显而易见：代码冗余、维护困难、容易出错

VGG 原论文中提出了一种非常优雅的思路：**用一个配置列表描述网络结构，用一个通用函数解析这个列表并生成网络**。这就是 `make_layers` 的核心思想

这个思想本质上是一种**数据驱动的网络构建（Data-driven Architecture Definition）**，它启发了后来几乎所有深度学习框架中的网络配置机制（如 YAML 配置文件、NAS 搜索空间等）

```python
# ════════════════════════════════════════════════════════════
# 第一部分：配置表
# ════════════════════════════════════════════════════════════

cfgs = {
    # cfgs 是一个字典（dict），键是字符串（网络名），值是列表（网络结构描述）
    # 
    # 列表里只有两种元素：
    #   整数（如 64）：表示"这里放一个卷积层，输出通道数是这个整数"
    #   字符串 'M'   ：表示"这里放一个 MaxPool 层"
    #
    # 列表从左到右，就是网络从输入到输出的顺序
    
    'vgg16': [
        64,  64,       'M',   # Block1：2个卷积（输出64通道）+ 1个MaxPool
        128, 128,      'M',   # Block2：2个卷积（输出128通道）+ 1个MaxPool
        256, 256, 256, 'M',   # Block3：3个卷积（输出256通道）+ 1个MaxPool
        512, 512, 512, 'M',   # Block4：3个卷积（输出512通道）+ 1个MaxPool
        512, 512, 512, 'M',   # Block5：3个卷积（输出512通道）+ 1个MaxPool
    ],
}

# ════════════════════════════════════════════════════════════
# 第二部分：make_layers 函数
# 作用：读取配置列表，逐个元素翻译成真正的 PyTorch 层对象
# ════════════════════════════════════════════════════════════

def make_layers(cfg: list, batch_norm: bool = False) -> nn.Sequential:
    # cfg       ：传入一个配置列表，比如 cfgs['vgg16']
    # batch_norm：是否加 BatchNorm 层，默认 False（不加）
    #             调用时写 make_layers(cfg, batch_norm=True) 就会加
    # 返回值    ：一个 nn.Sequential 对象（可以理解为"层的有序容器"）

    layers = []
    # layers 是一个普通的 Python 列表，用来临时存放每一层对象
    # 最终会把这个列表打包成 nn.Sequential
    # 初始为空列表 []，后续用 += 或 append 往里追加层

    in_channels = 3
    # in_channels 记录"当前层的输入通道数"
    # 第一个卷积层的输入是原始 RGB 图像，所以初始值是 3
    # 每次创建完一个卷积层之后，in_channels 会被更新为该层的输出通道数
    # 这样下一个卷积层创建时，就能知道自己的输入通道数是多少

    for v in cfg:
        # 遍历配置列表里的每一个元素
        # 第1次循环：v = 64  （vgg16 的第一个卷积层，输出通道64）
        # 第2次循环：v = 64  （第二个卷积层）
        # 第3次循环：v = 'M' （第一个MaxPool）
        # 第4次循环：v = 128 （第三个卷积层，输出通道128）
        # ...以此类推

        if v == 'M':
            # 如果当前元素是字符串 'M'，说明这里要放一个 MaxPool 层
            # kernel_size=2, stride=2 表示：用2×2的窗口，每次移动2格
            # 效果：输出的 H 和 W 都变成输入的一半
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            # layers += [某个对象] 等价于 layers.append(某个对象)
            # 这里用 += 是因为右边是一个列表 [...]，两个列表拼接

        else:
            # 如果当前元素不是 'M'，那它就是一个整数（比如 64、128、256、512）
            # 这个整数就是这一层卷积的输出通道数

            # 第一步：创建卷积层对象
            conv2d = nn.Conv2d(
                in_channels,   # 输入通道数（动态跟踪，初始为3，之后自动更新）
                v,             # 输出通道数（就是配置列表里这个整数，如 64）
                kernel_size=3, # 卷积核大小，固定 3×3
                padding=1      # 边缘补零1圈，保证卷积后 H 和 W 不变
            )
            # 此时 conv2d 只是一个层对象，还没有追加到 layers 里

            # 第二步：根据 batch_norm 决定追加哪些层
            if batch_norm:
                # batch_norm 是 True 时进这个分支
                # 追加三个层：卷积层 → BN层 → ReLU激活
                # nn.BatchNorm2d(v)：对输出的 v 个通道分别做归一化
                #   v 必须等于卷积层的输出通道数，两者要匹配
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                # batch_norm 是 False 时进这个分支（默认情况）
                # 追加两个层：卷积层 → ReLU激活
                # 没有 BN，就是原始 VGG 的设计
                layers += [conv2d, nn.ReLU(inplace=True)]

            # 第三步：更新 in_channels
            # 这一层的输出通道数是 v，下一层的输入通道数就是 v
            # 所以把 in_channels 更新为 v，供下一次循环使用
            in_channels = v
            # 举例：
            # 第1次循环 v=64：创建完 Conv(3→64) 后，in_channels 更新为 64
            # 第2次循环 v=64：创建 Conv(64→64)，in_channels 更新为 64
            # 第3次循环 v='M'：进入 if v=='M' 分支，in_channels 不变（还是64）
            # 第4次循环 v=128：创建 Conv(64→128)，in_channels 更新为 128

    # 循环结束后，layers 列表里装了所有层对象
    # 举例：vgg16 的 layers 列表最终会有：
    #   Conv(3→64), ReLU,
    #   Conv(64→64), ReLU,
    #   MaxPool,
    #   Conv(64→128), ReLU,
    #   Conv(128→128), ReLU,
    #   MaxPool,
    #   ... 共 31 个元素（13个卷积+13个ReLU+5个MaxPool）

    return nn.Sequential(*layers)
    # nn.Sequential(*layers) 的意思：
    # *layers 是 Python 的解包语法，把列表展开为一个个独立参数
    # 等价于 nn.Sequential(层1, 层2, 层3, ..., 层31)
    # nn.Sequential 会把这些层按顺序打包，调用时数据会依次经过每一层
```











