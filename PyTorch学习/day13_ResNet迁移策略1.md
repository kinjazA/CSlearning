# 01 迁移学习

## 01.1 背景

在深度学习中，训练一个性能良好的卷积神经网络（CNN）需要满足两个基本条件：

1. **海量带标注的数据**：ImageNet 拥有约 120 万张标注图像，这是大多数研究者和工程师无法自行收集的规模。
2. **庞大的计算资源**：在多块高端 GPU 上，训练一个 ResNet-50 仍需数天到数周。

迁移学习（Transfer Learning）的核心思想是：**已经在大规模数据集上充分训练好的网络，其内部参数所蕴含的"知识"（即对图像特征的抽象能力）是可以复用的**。不必从零开始，而是站在巨人的肩膀上

### ResNet 学到了什么？

理解迁移学习，首先要理解 ResNet 各层究竟"学到"了什么。

研究者通过可视化发现，卷积神经网络各层所学习的特征呈现出清晰的**层次结构**：

- **浅层（第1～2层卷积）**：学习最基础的视觉元素，如边缘（水平边缘、垂直边缘、斜边缘）、颜色对比块、简单纹理。这些特征极其通用，几乎对所有视觉任务都有价值

- **中间层（第3～4层）**：将浅层特征组合成更复杂的结构，如圆弧、纹理图案、局部形状（眼睛轮廓、车轮形状等）

- **深层（靠近输出层）**：学习高度抽象、任务特定的语义特征，如"狗脸的整体结构"、"汽车的侧面轮廓"。这些特征与 ImageNet 的类别高度绑定，迁移到差异较大的任务时，参考价值有限

   

这一规律是三种迁移学习策略设计的根本依据：

> * pytorch官方tutorial中写到：
>
>    These two major transfer learning scenarios look as follows:
>
>    - **Finetuning the ConvNet**: Instead of random initialization, we initialize the network with a pretrained network, like the one that is trained on imagenet 1000 dataset. Rest of the training looks as usual.
>    - **ConvNet as fixed feature extractor**: Here, we will freeze the weights for all of the network except that of the final fully connected layer. This last fully connected layer is replaced with a new one with random weights and only this layer is trained.

## 01.2 策略一之特征提取器

做法：把预训练模型的所有卷积层全部**冻结**，意思是在训练过程中，这些层的权重**完全不更新**，梯度不会回传到这些层。只把最后的 FC 分类头替换成新的（10 个类别的输出），并且**只训练这个新的 FC 头**

## 01.3 最终结果

![21](day13_ResNet%E8%BF%81%E7%A7%BB%E7%AD%96%E7%95%A51.assets/21-1773671823919-6.png)

![22](day13_ResNet%E8%BF%81%E7%A7%BB%E7%AD%96%E7%95%A51.assets/22-1773671823920-7.png)

##01.4 代码整体架构

整个项目的代码按照一条清晰的流水线组织，从上到下依次完成六个阶段

```css
main()
  │
  ├── 1. build_dataloaders()     ← 数据准备：下载 CIFAR-10、做增强、切分训练/验证/测试集
  │
  ├── 2. build_model()           ← 模型构建：加载预训练 ResNet34，做两处结构适配
  │
  ├── 3. apply_strategy()        ← 策略核心：冻结 backbone，只放开 stem 和 FC 训练
  │
  ├── 4. run_training()          ← 训练主循环：带早停、学习率调度、最优模型保存
  │
  ├── 5. evaluate() / get_predictions()  ← 测试集评估
  │
  └── 6. plot_*() / save_summary()       ← 可视化与报告生成
```

## 01.5 全局配置 `CFG`

代码最开头定义了一个字典 `CFG`，集中管理所有超参数。把"可能要调的东西"都放在一个地方，而不是散落在代码各处

```python
CFG = {
    "backbone":       "resnet34",     # 使用哪个预训练模型
    "num_classes":    10,             # CIFAR-10 有 10 个类别
    "batch_size":     128,            # 每次喂给模型多少张图
    "epochs":         20,             # 最多训练多少轮
    "lr":             1e-3,           # 学习率（FC 头是随机初始化的，可以用较大的学习率）
    "patience":       8,              # 早停耐心值：连续 8 轮没提升就停
    ...
}
```

# 02 数据准备

## 02.1 为什么要做数据增强？

深度学习模型容易"过拟合"——在训练集上表现很好，但换一张没见过的图就不行了。数据增强的作用是：**在训练时对图片做随机变换（裁剪、翻转、调色等），人为制造"新"样本，让模型见多识广**。

## 02.2 增强流水线详解

```python
train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),          # ①
    transforms.RandomHorizontalFlip(p=0.5),        # ②
    transforms.ColorJitter(brightness=0.2, ...),   # ③
    transforms.ToTensor(),                         # ④
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),  # ⑤
    transforms.RandomErasing(p=0.15, ...),         # ⑥
])
```

逐行解释：

- **① RandomCrop**：先在图片四周填充 4 个像素，然后随机裁剪回 32×32。效果是让图片内容在画面中"随机平移"，迫使模型不依赖物体的绝对位置。
- **② RandomHorizontalFlip**：50% 概率水平翻转图片。因为"一只朝左的猫"和"一只朝右的猫"都是猫。
- **③ ColorJitter**：随机微调亮度、对比度、饱和度，模拟不同光照条件。
- **④ ToTensor**：把 PIL 图片转为 PyTorch 张量，像素值从 \([0, 255]\) 的整数变为 \([0, 1]\) 的浮点数。
- **⑤ Normalize**：用 ImageNet 的均值和标准差做标准化。**这一步极其重要**——因为预训练模型是在 ImageNet 归一化后的数据上训练的，如果你喂进去的数据分布不一样，预训练特征就会失效。
- **⑥ RandomErasing**：以 15% 的概率随机遮挡图片的一小块区域，强迫模型不依赖某个局部特征。

**验证集和测试集不做随机增强**，只做 ToTensor + Normalize，因为评估时要看模型在"干净"数据上的真实表现。

## 02.3 数据集切分

```python
n_val   = int(n_total * 0.1)   # 从训练集中拿出 10% 当验证集
n_train = n_total - n_val
train_sub, val_sub = random_split(full_train, [n_train, n_val], ...)
```

CIFAR-10 原始数据中只有"训练集"和"测试集"两部分。代码额外从训练集里切出 10% 作为**验证集**，用于在训练过程中监控模型表现，防止过拟合。三者的分工是：

- **训练集**：模型从中学习参数
- **验证集**：每个 epoch 结束后检查模型表现，选出"最优模型"
- **测试集**：训练全部结束后，做一次最终评估，衡量真实泛化能力

# 03 模型构建与适配（`build_model`）

这是整个代码中最关键的部分之一。需要把一个"为 ImageNet 设计的模型"改造成"能处理 CIFAR-10 的模型"。改造涉及两个地方

## 03.1 加载预训练权重

```python
model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
```

这一行做了两件事：创建一个 ResNet34 的网络结构，然后把在 ImageNet 上训练好的参数加载进去`torchvision.models` 提供了开箱即用的预训练模型

## 03.2 适配一：修改 Stem 层

**问题**：ResNet 原本是为 224×224 的 ImageNet 图片设计的。它的第一层（stem）使用 7×7 大卷积核、步长为 2，后面还跟一个 MaxPool（步长 2）。两步下来，特征图尺寸缩小为原来的 \($\frac{1}{4}$\)。对于 224×224 的图来说，缩到 56×56 没问题；但 CIFAR-10 的图片只有 32×32，经过同样操作会缩到 8×8，大量空间信息丢失。

**解决方案**：

```python
# 把 7×7、步长2 的大卷积换成 3×3、步长1 的小卷积
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

# 去掉 MaxPool（用 Identity 替代，即"什么都不做"）
model.maxpool = nn.Identity()
```

这样修改后，输入 32×32 的图片，经过 stem 层后仍然是 32×32，空间信息被完整保留。

**权重复用技巧**：新的 conv1 是 3×3 卷积，但旧的 conv1 是 7×7 卷积。代码并没有直接用随机数初始化新的 conv1，而是取了旧 7×7 卷积核的**中心 3×3 区域**来初始化：

```python
old_conv1_weight = model.conv1.weight.data.clone()  # 保存旧权重 [64, 3, 7, 7]
# ... 创建新 conv1 后 ...
model.conv1.weight.copy_(old_conv1_weight[:, :, 2:5, 2:5])  # 取中心 3×3
```

==这背后的道理是：卷积核的中心区域通常包含最核心的特征模式，用它来初始化比完全随机初始化更稳定、收敛更快==

## 03.3 适配二：替换分类头（FC 层）

```python
in_features = model.fc.in_features   # ResNet34 的最后一层输出 512 维特征
model.fc = nn.Sequential(
    nn.Dropout(p=0.3),                         # 30% 的概率随机丢弃神经元，防过拟合
    nn.Linear(in_features, cfg["num_classes"])  # 512 → 10（CIFAR-10 有 10 类）
)
```

原始 ResNet34 的全连接层是 `Linear(512, 1000)`（输出 1000 个 ImageNet 类别的概率）。把它替换成 `Linear(512, 10)`，因为 CIFAR-10 只有 10 个类别。前面加了一层 Dropout，这是一种正则化手段，训练时随机让 30% 的神经元"休息"，防止模型过度依赖某些特定的神经元

## 03.4 小结

| 模块              | 原始用途             | 修改内容                       | 修改原因           |
| :---------------- | :------------------- | :----------------------------- | :----------------- |
| `conv1`           | 7×7 卷积，步长 2     | 改为 3×3 卷积，步长 1          | 避免小图过度下采样 |
| `maxpool`         | 3×3 最大池化，步长 2 | 改为 Identity（跳过）          | 同上               |
| `layer1`~`layer4` | 卷积残差块           | **不修改**                     | 保留预训练特征     |
| `fc`              | Linear(512, 1000)    | 改为 Dropout + Linear(512, 10) | 适配新的类别数     |

# 04 策略核心：冻结与解冻

## 04.1 什么是"冻结参数"？

在 PyTorch 中，每个参数（`nn.Parameter`）都有一个属性叫 `requires_grad`。当它为 `True` 时，该参数会参与梯度计算和反向传播，也就是"会被训练"；当它为 `False` 时，该参数在训练中保持不变，也就是"被冻结"了。

## 04.2 冻结逻辑详解

```python
def apply_strategy(model):
    # 第一步：先把所有参数都冻结
    for param in model.parameters():
        param.requires_grad = False

    # 第二步：有选择地解冻需要训练的部分

    # 解冻新的 conv1（因为结构已经被改了，必须重新训练）
    for param in model.conv1.parameters():
        param.requires_grad = True

    # 解冻 bn1（Batch Normalization 层，配合 conv1 一起训练更稳定）
    for param in model.bn1.parameters():
        param.requires_grad = True

    # 解冻分类头（这是全新的层，必须训练）
    for param in model.fc.parameters():
        param.requires_grad = True
```

==**为什么 conv1 也要解冻？** 前面我们把 conv1 从 7×7 改成了 3×3，虽然用了旧权重的中心区域做初始化，但结构毕竟变了。如果继续冻结它，就等于让一个"没有完全适配的层"永远不被调整，这会拖累整个模型的表现==

**为什么 bn1 也要解冻？** BatchNorm 层的作用是对输入数据做归一化处理。它紧跟在 conv1 后面，如果 conv1 的输出分布变了（因为结构和权重都变了），但 bn1 还是用旧的统计量，就会产生"分布不匹配"的问题。

**最终效果**：整个 ResNet34 大约有 2100 万个参数，但实际参与训练的只有 conv1 + bn1 + fc 这一小部分（大约占总参数量的 5% 左右）。其余 95% 的参数保持冻结，保留了 ImageNet 上学到的特征提取能力。

## 04.3 `requires_grad` 的工作机制

用一个简单的类比来理解：

想象模型是一条流水线，从输入到输出有很多道工序（层）。反向传播就像"回过头来检查每道工序做得好不好，然后调整"。当你把某些工序的 `requires_grad` 设为 `False`，就相当于告诉系统："这些工序不用检查、不用调整了。"这样做有两个好处：**省计算量**（不用算这些层的梯度了），**防止破坏**（预训练好的特征不会被破坏）。

# 05 训练流程详解

## 05.1 损失函数：带标签平滑的交叉熵

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

交叉熵损失是分类任务的标配损失函数。`label_smoothing=0.1` 是一个正则化技巧——它把"硬标签"（比如 \([0,0,0,1,0,0,0,0,0,0]\)，只有正确类别是 1）变成"软标签"（比如 \([0.01, 0.01, 0.01, 0.91, 0.01, ...]\)）。好处是防止模型过于自信，提高泛化能力

## 05.2 优化器：AdamW

```python
optimizer = optim.AdamW(trainable, lr=1e-3, weight_decay=1e-4)
```

这里传入的是 `trainable`（只包含 `requires_grad=True` 的参数），而不是 `model.parameters()`。这确保优化器只更新我们想要训练的那部分参数。

`AdamW` 是 Adam 优化器的改进版本，`weight_decay=1e-4` 是 L2 正则化系数，防止参数值过大。

## 05.3 学习率调度：OneCycleLR

```python
scheduler = OneCycleLR(
    optimizer, max_lr=1e-3,
    steps_per_epoch=len(train_loader), epochs=n_epochs,
    pct_start=0.3, anneal_strategy="cos",
)
```

OneCycleLR 是一种"先升后降"的学习率策略：

1. **前 30% 的训练过程**（`pct_start=0.3`）：学习率从很小逐渐升到最大值 $(10^{-3})$
2. **后 70% 的训练过程**：学习率按余弦曲线（`anneal_strategy="cos"`）缓慢下降到接近 0

这种"warm-up + cosine decay"的策略被大量实验证明比固定学习率效果更好

## 05.4 早停机制

```python
if vl_acc > best_acc:
    best_acc = vl_acc
    best_wts = copy.deepcopy(model.state_dict())  # 保存当前最优权重
    no_improve = 0
else:
    no_improve += 1

if no_improve >= cfg["patience"]:   # patience=8
    print("早停：连续 8 epoch 无提升，终止训练")
    break
```

早停的逻辑：每个 epoch 结束后看验证集准确率有没有创新高。如果连续 8 个 epoch 都没有提升，就判定模型"已经学不到新东西了"，提前终止训练。同时，代码始终保留着"历史最优"的模型权重，最后恢复为这个最优版本

## 05.5 梯度裁剪

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

这行代码在每次反向传播后执行，作用是：如果某一步的梯度"爆炸"（数值特别大），就把它缩放到不超过 1.0。这是一种训练稳定性保障，防止因为某个异常样本导致参数被更新到离谱的值

# 06 核心 API

## 06.1 加载预训练模型

```python
models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
```

`torchvision.models` 提供了各种经典网络结构的实现。`weights` 参数指定要加载哪组预训练权重。`IMAGENET1K_V1` 表示在 ImageNet-1K 上训练的第一个版本的权重。如果写 `weights=None` 则只创建网络结构而不加载权重

## 06.2 冻结参数

```python
param.requires_grad = False
```

这是 PyTorch 中控制参数是否参与训练的核心开关。设为 `False` 后，该参数在前向传播时仍然正常参与计算（作为固定的常数），但反向传播时不会为它计算梯度，优化器也不会更新它

## 06.3 替换模型的子模块

```python
model.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(512, 10)
)
model.maxpool = nn.Identity()
```

PyTorch 的模型是由 `nn.Module` 组成的树形结构。任何子模块都可以通过直接赋值来替换。`nn.Identity()` 是一个"恒等映射"——输入是什么，输出就是什么——常用来"跳过"原有模块

## 06.4 `model.train()` 与 `model.eval()`

```python
model.train()   # 训练模式
model.eval()    # 评估模式
```

这两个方法的作用是切换模型中 Dropout 和 BatchNorm 的行为：

- **训练模式** (`train()`): Dropout 生效（随机丢弃神经元），BatchNorm 用当前 batch 的统计量
- **评估模式** (`eval()`): Dropout 关闭（所有神经元参与计算），BatchNorm 用训练期间积累的全局统计量

注意：`model.eval()` **不等于** `requires_grad = False`。前者只改变 Dropout/BN 的行为，后者才是控制梯度计算的开关

## 06.5 `@torch.no_grad()` 装饰器

```python
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    ...
```

这个装饰器告诉 PyTorch："在这个函数内部，不需要记录任何梯度信息。"因为评估时不需要反向传播，关掉梯度记录可以**节省显存**和**加快推理速度**

## 06.6 保存与加载模型权重

```python
# 保存
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "val_acc": best_acc
}, "best_model.pth")

# 加载
model.load_state_dict(best_wts)
```

`model.state_dict()` 返回一个字典，里面是模型所有参数的名字和对应的张量。保存时把它打包成 `.pth` 文件；加载时用 `load_state_dict()` 把参数灌回模型。`copy.deepcopy()` 是 Python 的深拷贝，确保保存的是当时的参数快照，而不是一个会随后续训练被修改的引用

## 06.7 `DataLoader` 与 `random_split`

```python
train_sub, val_sub = random_split(full_train, [n_train, n_val], generator=...)
train_loader = DataLoader(train_sub, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
```

`random_split` 把一个数据集随机切成若干份。`DataLoader` 是 PyTorch 数据加载的核心工具，它负责把数据集切分成 batch、打乱顺序（`shuffle`）、多进程并行加载（`num_workers`）。`pin_memory=True` 是 GPU 训练的优化项，让数据先锁定在内存中，加速 CPU→GPU 的传输

# 07 关键设计决策总结

## 07.1 为什么用 ImageNet 的归一化参数？

预训练模型是在 ImageNet 归一化后的数据上训练的，它的每一层权重都"适应"了那种数据分布。如果用 CIFAR-10 自己的均值和方差做归一化，输入的数据分布就和预训练时不一样了，预训练的特征提取能力就会大打折扣

## 07.2 为什么 FC 头可以用较大的学习率（1e-3）？

FC 头是全新的、随机初始化的层，它的参数还处于"一无所知"的状态，需要较大的学习率快速学到东西。而如果要微调 backbone（Fine-Tuning 策略），backbone 的参数已经很接近最优解了，用大学习率反而会破坏已有知识，通常需要小 10 倍甚至 100 倍的学习率

## 07.3 为什么要用 `copy.deepcopy` 保存最优模型？

`model.state_dict()` 返回的字典里存的是张量的**引用**，不是副本。如果你不做深拷贝就直接保存，后续训练修改了模型参数时，之前"保存"的也会被改掉。深拷贝确保拿到的是一个独立的快照

## 07.4 Dropout 的位置为什么在 FC 之前？

Dropout 放在全连接层之前，作用于 512 维的特征向量。它随机丢弃 30% 的特征，迫使模型不依赖任何单一特征做决策，从而提高泛化能力。放在 FC 之后就没意义了——输出都 10 维了，再丢弃反而破坏分类结果

# 08 进阶思考

## 08.1 Feature Extractor 的局限

这种策略的缺点也很明显：backbone 完全冻结意味着它无法针对目标任务做任何调整。如果你的目标数据和 ImageNet 差异较大（比如医学影像、卫星图），预训练特征可能不够好用，此时应该考虑 Fine-Tuning（微调 backbone）

## 08.2 从 Feature Extractor 到 Fine-Tuning

如果想更进一步，可以把 `apply_strategy` 中冻结的层逐步解冻：先只解冻 `layer4`，再解冻 `layer3`……这就是所谓的"逐层微调"（Gradual Unfreezing）。每多解冻一层，模型的表达能力更强，但也需要更多数据和更谨慎的学习率设置

# 09 完整代码

```python
"""
================================================================================
  迁移学习策略一：Feature Extractor（特征提取器）
================================================================================
  核心思想：
    冻结预训练 ResNet 的所有卷积层，只训练新增的分类头（FC层）。
    预训练模型作为静态特征提取器，不参与梯度更新。

  适用场景：
    - 目标数据集较小（< 5000 张）
    - 目标域与 ImageNet 高度相似
    - 快速验证任务可行性，获取 baseline

  运行方式：
    python strategy1_feature_extractor.py

  依赖：
    pip install torch torchvision tqdm matplotlib scikit-learn
================================================================================
"""

import os
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, random_split

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np
import matplotlib
matplotlib.use("Agg")          # 无显示器环境下使用非交互后端
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm


# ============================================================
#  全局配置
# ============================================================
CFG = {
    "strategy_name":  "Feature Extractor",
    "data_dir":       "./data",
    "save_dir":       "./results/strategy1_feature_extractor",
    "backbone":       "resnet34",
    "num_classes":    10,
    "batch_size":     128,
    "num_workers":    2,
    "val_split":      0.1,
    "seed":           42,
    "device":         "cuda" if torch.cuda.is_available() else "cpu",
    # 训练超参数
    "epochs":         20,
    "lr":             1e-3,       # FC 头随机初始化，可以用较大学习率
    "weight_decay":   1e-4,
    "label_smoothing": 0.1,
    # 早停
    "patience":       8,
}

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

# ImageNet 归一化参数（必须与预训练时一致）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

torch.manual_seed(CFG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG["seed"])
os.makedirs(CFG["save_dir"], exist_ok=True)


# ============================================================
#  1. 数据准备
# ============================================================
def build_transforms():
    """构建数据增强流水线，归一化参数必须使用 ImageNet 的统计量"""
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.1)),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


def build_dataloaders(cfg):
    train_tf, val_tf = build_transforms()
    full_train = torchvision.datasets.CIFAR10(
        root=cfg["data_dir"], train=True, download=True, transform=train_tf)

    n_total = len(full_train)
    n_val   = int(n_total * cfg["val_split"])
    n_train = n_total - n_val
    train_sub, val_sub = random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg["seed"]))

    # 验证集单独加载以使用 val_tf（无随机增强）
    full_val = torchvision.datasets.CIFAR10(
        root=cfg["data_dir"], train=True, download=False, transform=val_tf)
    val_dataset = torch.utils.data.Subset(full_val, val_sub.indices)

    test_dataset = torchvision.datasets.CIFAR10(
        root=cfg["data_dir"], train=False, download=True, transform=val_tf)

    kw = dict(num_workers=cfg["num_workers"], pin_memory=True)
    train_loader = DataLoader(train_sub,    batch_size=cfg["batch_size"], shuffle=True,  **kw)
    val_loader   = DataLoader(val_dataset,  batch_size=cfg["batch_size"]*2, shuffle=False, **kw)
    test_loader  = DataLoader(test_dataset, batch_size=cfg["batch_size"]*2, shuffle=False, **kw)

    print(f"[数据] 训练: {n_train}  验证: {n_val}  测试: {len(test_dataset)}")
    return train_loader, val_loader, test_loader


# ============================================================
#  2. 模型构建与适配
# ============================================================
def build_model(cfg):
    """
    加载预训练 ResNet34 并做两处 CIFAR-10 适配：
      1. 修改 stem：Conv(7×7,s2) → Conv(3×3,s1)，移除 MaxPool
         避免 32×32 图像在第一层就被过度下采样
      2. 替换 FC 头：1000 类 → 10 类

    工程上更常用的做法：
      - 新 conv1 不用随机初始化后直接冻结
      - 而是尽量复用预训练 conv1 的权重做初始化
    """
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

    # 先保存原始预训练 conv1 权重: [64, 3, 7, 7]
    old_conv1_weight = model.conv1.weight.data.clone()

    # 适配 1：stem 层修改
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # 用原 7x7 卷积核的中心 3x3 区域初始化新 conv1
    # 这是迁移学习里很常见的做法，比随机初始化更稳
    with torch.no_grad():
        model.conv1.weight.copy_(old_conv1_weight[:, :, 2:5, 2:5])

    model.maxpool = nn.Identity()

    # 适配 2：替换分类头
    in_features = model.fc.in_features      # ResNet34 → 512
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, cfg["num_classes"])
    )
    return model



# ============================================================
#  3. 策略核心：冻结配置
# ============================================================
def apply_strategy(model):
    """
    策略一的工程化修正版本：
      - 冻结大部分预训练 backbone
      - 由于 conv1 被改了结构，不能继续冻结
      - 放开训练：conv1 + bn1 + fc
      - 其余层保持冻结

    这样既保留“Feature Extractor”的主体思想，
    又避免“新第一层随机初始化但不训练”的问题。
    """
    # 先冻结全部参数
    for param in model.parameters():
        param.requires_grad = False

    # 新 stem 需要训练
    for param in model.conv1.parameters():
        param.requires_grad = True

    # 通常 bn1 也一起解冻，让输入分布适配更稳定
    for param in model.bn1.parameters():
        param.requires_grad = True

    # 分类头继续训练
    for param in model.fc.parameters():
        param.requires_grad = True

    print("\n[策略] Feature Extractor（修正版）—— 冻结 backbone，仅训练 stem + FC")
    _print_param_status(model)
    return model



def _print_param_status(model):
    """打印各子模块的冻结状态"""
    print(f"  {'模块':<10} {'状态':<12} {'参数量':>10}")
    print("  " + "-" * 36)
    for name, module in model.named_children():
        params    = list(module.parameters())
        if not params:
            continue
        total     = sum(p.numel() for p in params)
        trainable = sum(p.numel() for p in params if p.requires_grad)
        status    = "✓ 训练" if trainable == total else ("✗ 冻结" if trainable == 0 else "部分")
        print(f"  {name:<10} {status:<12} {total:>10,}")
    total_all  = sum(p.numel() for p in model.parameters())
    train_all  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {'合计':<10} {train_all:>10,} / {total_all:>10,} 可训练 "
          f"({100*train_all/total_all:.1f}%)\n")


# ============================================================
#  4. 训练与评估
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, scheduler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="  Train", leave=False, ncols=90)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
        pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100*correct/total:.1f}%")
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out  = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def get_predictions(model, loader, device):
    """返回所有预测标签和真实标签，用于混淆矩阵等可视化"""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        out  = model(imgs)
        probs = torch.softmax(out, dim=1)
        all_preds.append(out.argmax(1).cpu())
        all_labels.append(labels)
        all_probs.append(probs.cpu())
    return (torch.cat(all_preds).numpy(),
            torch.cat(all_labels).numpy(),
            torch.cat(all_probs).numpy())


def run_training(model, train_loader, val_loader, cfg):
    """带早停和最优模型保存的训练主循环"""
    device    = cfg["device"]
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    n_epochs  = cfg["epochs"]

    # 只优化可训练参数（FC 头）
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = OneCycleLR(
        optimizer, max_lr=cfg["lr"],
        steps_per_epoch=len(train_loader), epochs=n_epochs,
        pct_start=0.3, anneal_strategy="cos",
    )

    history   = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_acc  = 0.0
    best_wts  = copy.deepcopy(model.state_dict())
    no_improve = 0

    print(f"\n{'='*60}")
    print(f"  开始训练：{cfg['strategy_name']}  共 {n_epochs} epochs")
    print(f"  设备: {device.upper()}")
    print(f"{'='*60}")

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, scheduler, device)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        flag = ""
        if vl_acc > best_acc:
            best_acc = vl_acc
            best_wts = copy.deepcopy(model.state_dict())
            torch.save({"epoch": epoch, "model_state_dict": best_wts,
                        "val_acc": best_acc},
                       os.path.join(cfg["save_dir"], "best_model.pth"))
            flag = "  ← best"
            no_improve = 0
        else:
            no_improve += 1

        print(f"  Epoch [{epoch:3d}/{n_epochs}] "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc*100:.2f}%  "
              f"val_loss={vl_loss:.4f} val_acc={vl_acc*100:.2f}%  "
              f"({elapsed:.1f}s){flag}")

        if no_improve >= cfg["patience"]:
            print(f"\n  早停：连续 {cfg['patience']} epoch 无提升，终止训练")
            break

    model.load_state_dict(best_wts)
    print(f"\n  最优验证集准确率: {best_acc*100:.2f}%")
    return model, history


# ============================================================
#  5. 可视化
# ============================================================
def plot_training_curves(history, cfg):
    """绘制训练/验证的 Loss、Accuracy、学习率曲线"""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"训练曲线 — {cfg['strategy_name']}", fontsize=14, fontweight="bold")

    # Loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train Loss", color="#2196F3", linewidth=2)
    ax.plot(epochs, history["val_loss"],   label="Val Loss",   color="#FF5722", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Loss 曲线"); ax.legend(); ax.grid(alpha=0.3)

    # Accuracy
    ax = axes[1]
    tr_acc = [a * 100 for a in history["train_acc"]]
    vl_acc = [a * 100 for a in history["val_acc"]]
    ax.plot(epochs, tr_acc, label="Train Acc", color="#2196F3", linewidth=2)
    ax.plot(epochs, vl_acc, label="Val Acc",   color="#FF5722", linewidth=2)
    best_ep  = int(np.argmax(history["val_acc"])) + 1
    best_val = max(vl_acc)
    ax.axvline(x=best_ep, color="gray", linestyle="--", alpha=0.6, label=f"Best epoch={best_ep}")
    ax.scatter([best_ep], [best_val], color="#FF5722", zorder=5, s=80)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy 曲线"); ax.legend(); ax.grid(alpha=0.3)

    # 学习率
    ax = axes[2]
    ax.plot(epochs, history["lr"], color="#4CAF50", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
    ax.set_title("学习率变化"); ax.grid(alpha=0.3)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    plt.tight_layout()
    path = os.path.join(cfg["save_dir"], "01_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[可视化] 训练曲线 → {path}")


def plot_confusion_matrix(y_true, y_pred, cfg):
    """绘制归一化混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(CIFAR10_CLASSES, fontsize=10)
    ax.set_xlabel("预测标签", fontsize=12); ax.set_ylabel("真实标签", fontsize=12)
    ax.set_title(f"混淆矩阵（归一化）— {cfg['strategy_name']}", fontsize=13, fontweight="bold")

    for i in range(10):
        for j in range(10):
            color = "white" if cm_norm[i, j] > 0.55 else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    plt.tight_layout()
    path = os.path.join(cfg["save_dir"], "02_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[可视化] 混淆矩阵 → {path}")


def plot_per_class_accuracy(y_true, y_pred, cfg):
    """绘制每个类别的准确率条形图"""
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100

    colors = plt.cm.RdYlGn(per_class_acc / 100)
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(CIFAR10_CLASSES, per_class_acc, color=colors, edgecolor="white", linewidth=0.8)

    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    overall_acc = np.mean(per_class_acc)
    ax.axhline(y=overall_acc, color="#2196F3", linestyle="--", linewidth=2,
               label=f"整体平均: {overall_acc:.1f}%")
    ax.set_ylim(0, 110)
    ax.set_xlabel("类别", fontsize=12); ax.set_ylabel("准确率 (%)", fontsize=12)
    ax.set_title(f"各类别准确率 — {cfg['strategy_name']}", fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(cfg["save_dir"], "03_per_class_accuracy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[可视化] 各类别准确率 → {path}")


def plot_sample_predictions(model, test_loader, cfg, n_samples=40):
    """
    从测试集中随机抽取样本，可视化预测结果。
    绿色边框=预测正确，红色边框=预测错误
    """
    device = cfg["device"]
    model.eval()

    images_list, labels_list, preds_list = [], [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu()
            images_list.append(imgs.cpu())
            labels_list.append(labels)
            preds_list.append(preds)
            if sum(len(x) for x in labels_list) >= n_samples:
                break

    images = torch.cat(images_list)[:n_samples]
    labels = torch.cat(labels_list)[:n_samples]
    preds  = torch.cat(preds_list)[:n_samples]

    # 反归一化，还原可视化用的图像
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    images = images * std + mean
    images = images.clamp(0, 1)

    cols = 8
    rows = n_samples // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 2.2))
    fig.suptitle(f"预测样本展示 — {cfg['strategy_name']}\n绿框=正确  红框=错误",
                 fontsize=12, fontweight="bold")

    for idx, ax in enumerate(axes.flat):
        img = images[idx].permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.axis("off")
        correct = preds[idx].item() == labels[idx].item()
        color   = "#4CAF50" if correct else "#F44336"
        for spine in ax.spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(3); spine.set_visible(True)
        ax.set_title(f"预: {CIFAR10_CLASSES[preds[idx]]}\n真: {CIFAR10_CLASSES[labels[idx]]}",
                     fontsize=7, color=color, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(cfg["save_dir"], "04_sample_predictions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[可视化] 样本预测 → {path}")


def plot_top_errors(y_true, y_pred, y_prob, cfg, top_n=10):
    """
    可视化模型最常犯的 top-N 错误类别对。
    帮助分析模型在哪些类别间容易混淆。
    """
    error_pairs = {}
    for true, pred in zip(y_true, y_pred):
        if true != pred:
            key = (CIFAR10_CLASSES[true], CIFAR10_CLASSES[pred])
            error_pairs[key] = error_pairs.get(key, 0) + 1

    sorted_pairs = sorted(error_pairs.items(), key=lambda x: x[1], reverse=True)[:top_n]
    labels = [f"{p[0][0]} → {p[0][1]}" for p in sorted_pairs]
    counts = [p[1] for p in sorted_pairs]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = plt.cm.Reds(np.linspace(0.4, 0.9, len(counts)))
    bars    = ax.barh(labels[::-1], counts[::-1], color=colors[::-1], edgecolor="white")

    for bar, count in zip(bars, counts[::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("错误次数", fontsize=12)
    ax.set_title(f"Top-{top_n} 易混淆类别对 — {cfg['strategy_name']}", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(cfg["save_dir"], "05_top_errors.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[可视化] 易混淆类别 → {path}")


def save_summary(history, y_true, y_pred, cfg, test_acc):
    """保存文字版训练摘要报告"""
    report = classification_report(y_true, y_pred, target_names=CIFAR10_CLASSES)
    best_val = max(history["val_acc"]) * 100
    path = os.path.join(cfg["save_dir"], "summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\n")
        f.write(f"  训练策略: {cfg['strategy_name']}\n")
        f.write(f"  Backbone: {cfg['backbone']}\n")
        f.write(f"  训练 Epoch 数: {len(history['train_loss'])}\n")
        f.write(f"  最优验证集准确率: {best_val:.2f}%\n")
        f.write(f"  测试集准确率:     {test_acc*100:.2f}%\n")
        f.write(f"{'='*60}\n\n")
        f.write("分类报告（测试集）:\n")
        f.write(report)
    print(f"[摘要] 训练报告 → {path}")


# ============================================================
#  主程序
# ============================================================
def main():
    cfg    = CFG
    device = cfg["device"]
    print(f"\n{'='*60}")
    print(f"  策略一：{cfg['strategy_name']}")
    print(f"  设备: {device.upper()}" +
          (f"  ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    print(f"{'='*60}")

    # 1. 数据
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    # 2. 模型
    model = build_model(cfg).to(device)

    # 3. 应用策略（冻结 backbone）
    model = apply_strategy(model)

    # 4. 训练
    model, history = run_training(model, train_loader, val_loader, cfg)

    # 5. 测试集评估
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n[测试集] Loss={test_loss:.4f}  Accuracy={test_acc*100:.2f}%")

    # 6. 获取所有预测结果（用于可视化）
    y_pred, y_true, y_prob = get_predictions(model, test_loader, device)

    # 7. 可视化
    print("\n[可视化] 生成图表中...")
    plot_training_curves(history, cfg)
    plot_confusion_matrix(y_true, y_pred, cfg)
    plot_per_class_accuracy(y_true, y_pred, cfg)
    plot_sample_predictions(model, test_loader, cfg)
    plot_top_errors(y_true, y_pred, y_prob, cfg)
    save_summary(history, y_true, y_pred, cfg, test_acc)

    print(f"\n所有结果已保存至: {cfg['save_dir']}/")
    print("完成！")


if __name__ == "__main__":
    main()
```

