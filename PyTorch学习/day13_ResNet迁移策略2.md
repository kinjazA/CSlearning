# 01 迁移策略之——全量微调

本代码采用的是 **策略二：Full Fine-tuning（全网络微调）**，即在加载预训练权重后，最终解冻所有层，使用较小的学习率对整个网络做端到端训练。预训练权重此时充当"优质初始值"，而不是被固定不变的特征提取器

训练效果：

![image-20260319200949462](day13_ResNet%E8%BF%81%E7%A7%BB%E7%AD%96%E7%95%A52.assets/image-20260319200949462.png)

![image-20260319201009988](day13_ResNet%E8%BF%81%E7%A7%BB%E7%AD%96%E7%95%A52.assets/image-20260319201009988.png)

![image-20260319201029433](day13_ResNet%E8%BF%81%E7%A7%BB%E7%AD%96%E7%95%A52.assets/image-20260319201029433.png)

#02 代码整体架构

整个脚本可以分为以下六大模块，它们按流水线的顺序依次运行：

```css
┌──────────────────────────────────────────────────────────┐
│  全局配置 (CFG)                                          │
│  定义超参数、路径、设备等所有训练参数                      │
└──────────────┬───────────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────────┐
│  模块一：数据准备 (build_transforms / build_dataloaders)  │
│  下载 CIFAR-10 → 数据增强 → 划分训练/验证/测试集          │
└──────────────┬───────────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────────┐
│  模块二：模型构建 (build_model)                           │
│  加载预训练 ResNet34 → 修改输入层和分类头 → 适配 CIFAR-10 │
└──────────────┬───────────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────────┐
│  模块三：两阶段训练策略 (run_training)                    │
│  阶段一：冻结 backbone，预热 FC 头（8 epoch）             │
│  阶段二：解冻全网络，小学习率全量微调（30 epoch）          │
└──────────────┬───────────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────────┐
│  模块四：训练循环与评估                                   │
│  单 epoch 训练 / 验证评估 / 早停机制 / 最优模型保存        │
└──────────────┬───────────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────────┐
│  模块五：测试 + 可视化                                    │
│  训练曲线 / 混淆矩阵 / 各类别准确率 / 预测样本 / 易混淆对 │
└──────────────────────────────────────────────────────────┘
```

#03 模块详解

## 03.1 全局配置（CFG 字典）

所有超参数集中在一个字典 `CFG` 中管理，这是一个良好的工程实践——避免在代码各处散落"魔法数字"，方便统一调参。

```python
CFG = {
    "backbone":        "resnet34",      # 使用哪个预训练模型
    "num_classes":     10,              # CIFAR-10 共 10 类
    "batch_size":      128,             # 每批处理 128 张图
    "warmup_epochs":   8,               # 阶段一：预热 8 个 epoch
    "warmup_lr":       1e-3,            # 阶段一学习率（较大）
    "finetune_epochs": 30,              # 阶段二：微调 30 个 epoch
    "finetune_lr":     1e-4,            # 阶段二学习率（比阶段一小 10 倍）
    "weight_decay":    1e-4,            # L2 正则化
    "label_smoothing": 0.1,             # 标签平滑
    "patience":        10,              # 早停容忍度
    ...
}
```

几个关键参数的含义：

- **`warmup_lr` vs `finetune_lr`**：阶段一学习率 \(10^{-3}\) 比阶段二 \(10^{-4}\) 大 10 倍。这是因为阶段一只训练随机初始化的 FC 头，需要较大步长快速收敛；阶段二解冻了预训练权重，必须用小步长"微调"，否则会破坏已经学好的特征——这种现象叫做**灾难性遗忘（Catastrophic Forgetting）**。
- **`label_smoothing`**：标签平滑。普通的 one-hot 标签（比如 `[0,0,1,0,...,0]`）过于"自信"，标签平滑会把它变成类似 `[0.01, 0.01, 0.91, 0.01, ..., 0.01]`，迫使模型不要过度自信，从而提升泛化能力。
- **`patience`**：早停机制的容忍度。如果验证集准确率连续 10 个 epoch 没有提升，就提前终止训练，防止过拟合。

## 03.2 数据准备

### 03.2.1 数据增强（Data Augmentation）

数据增强是通过对原始图像施加随机变换来"虚拟扩充"训练集的技术。代码中使用了以下增强策略：

```python
train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),          # 随机裁剪
    transforms.RandomHorizontalFlip(p=0.5),        # 50% 概率水平翻转
    transforms.ColorJitter(brightness=0.2, ...),   # 颜色抖动
    transforms.ToTensor(),                          # 转为张量
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),  # ImageNet 标准化
    transforms.RandomErasing(p=0.15, ...),         # 随机擦除
])
```

每种变换的作用：

- **RandomCrop(32, padding=4)**：先在图像四周填充 4 个像素，然后随机裁剪回 32×32。效果是让图像在小范围内平移，增强模型对位置变化的鲁棒性。
- **RandomHorizontalFlip**：水平翻转。飞机、汽车翻转后仍然是飞机、汽车。
- **ColorJitter**：随机调整亮度、对比度、饱和度，模拟不同光照条件。
- **Normalize**：使用 ImageNet 的均值和标准差做归一化。这一步至关重要——因为预训练模型是在 ImageNet 上训练的，它"期望"输入数据的分布与 ImageNet 一致。如果用 CIFAR-10 自身的均值和标准差归一化，预训练权重的效果就会打折扣。
- **RandomErasing**：随机遮挡图像的一小块区域，迫使模型不要依赖局部特征，类似于 Dropout 的思想。

注意：验证集和测试集**不做任何随机增强**，只做 `ToTensor()` + `Normalize()`。因为评估时需要保持数据一致性。

### 03.2.2 数据集划分

```python
n_val   = int(n_total * 0.1)   # 10% 作为验证集
n_train = n_total - n_val      # 90% 用于训练
```

CIFAR-10 原始训练集有 50000 张图，代码将其拆分为 45000 张训练 + 5000 张验证。验证集用于在训练过程中监控模型表现、执行早停判断和选择最优模型。测试集（10000 张）则是完全独立的，只在最后用一次。

## 03.3 模型构建——如何"改装"预训练 ResNet34

这是整个代码中最关键的环节之一。`build_model()` 函数做了三件事：

```python
def build_model(cfg):
    # 第一步：加载 ImageNet 预训练的 ResNet34
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

    # 第二步：替换第一个卷积层（适配 CIFAR-10 的 32×32 小尺寸图像）
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")

    # 第三步：移除最大池化层
    model.maxpool = nn.Identity()

    # 第四步：替换分类头（1000 类 → 10 类）
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_features, cfg["num_classes"]))

    return model
```

**为什么要改这些层？** 逐一解释：

**（1）替换 `conv1`**

原始 ResNet34 的第一个卷积层是 `Conv2d(3, 64, kernel_size=7, stride=2, padding=3)`，设计用于 224×224 的 ImageNet 图像。7×7 的大卷积核配合 stride=2 会直接把特征图的空间尺寸缩小一半。但 CIFAR-10 的图像只有 32×32，经过这样的降采样后只剩 16×16，信息损失太大。

替换为 `kernel_size=3, stride=1, padding=1` 的卷积层后，空间尺寸保持不变（32×32 → 32×32），保留了更多细节信息。

注意：由于这一层被替换了，它的权重是**随机初始化**的（使用 Kaiming 初始化），而不是来自 ImageNet 的预训练权重。

**（2）移除 `maxpool`**

原始 ResNet34 在 `conv1` 后还有一个 3×3 的最大池化层（stride=2），会再次将空间尺寸减半。对 32×32 的小图像来说同样损失过大，所以用 `nn.Identity()` 替代——这是一个"什么都不做"的模块，等效于删除这一层。

**（3）替换分类头 `fc`**

原始 ResNet34 的最后一层是 `Linear(512, 1000)`（ImageNet 有 1000 类），需要改成 `Linear(512, 10)`（CIFAR-10 有 10 类）。此外还加了 `Dropout(p=0.3)`，在训练时随机丢弃 30% 的神经元输出，防止过拟合。

这一层同样是**随机初始化**的。

**总结模型结构的权重来源：**

| 层                       | 权重来源        | 说明                        |
| ------------------------ | --------------- | --------------------------- |
| `conv1`（替换后）        | 随机初始化      | 适配 32×32 输入             |
| `bn1`, `layer1`~`layer4` | ImageNet 预训练 | 提取通用视觉特征的 backbone |
| `maxpool`（替换后）      | 无参数          | Identity，直接跳过          |
| `fc`（替换后）           | 随机初始化      | 适配 10 类分类              |

# 04 核心策略：两阶段训练

为什么不直接解冻所有层开始训练，而要分两个阶段？

## 04.1 问题背景：梯度冲击

模型中存在两类权重：预训练的 backbone 权重（已经很好了）和随机初始化的 FC 头权重（一团乱）。如果一开始就同时训练所有层，FC 头由于权重随机，产生的 loss 很大，反向传播时梯度也很大。这些大梯度会经过链式法则传递到 backbone 各层，可能会"冲坏"已经训练好的特征提取器

## 04.2 阶段一：预热 FC 头（Warmup）

#### 实现方式——冻结 backbone

```python
def freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = False    # 先冻结所有层
    for param in model.fc.parameters():
        param.requires_grad = True     # 再单独解冻 FC 头
    # 同时解冻被替换的 conv1（它是随机初始化的，需要训练）
    for param in model.conv1.parameters():
        param.requires_grad = True
```

**`requires_grad` 是什么？** 这是 PyTorch 张量的一个属性，控制该张量是否参与梯度计算。设为 `False` 后：

1. 该参数在反向传播时不会计算梯度（节省显存和计算量）
2. 优化器不会更新该参数的值（即使它被加入了优化器的参数列表）

所以 `freeze_backbone()` 的效果是：整个网络中，只有 FC 头的参数（Dropout + Linear）会被训练，backbone 中约 2100 万个参数全部保持不变。

#### 训练配置

```python
opt1 = optim.AdamW(warmup_params, lr=1e-3, weight_decay=1e-4)
sch1 = OneCycleLR(opt1, max_lr=1e-3, steps_per_epoch=len(train_loader),
                  epochs=8, pct_start=0.3)
```

- **优化器只接收可训练参数**：`warmup_params = [p for p in model.parameters() if p.requires_grad]`，只把 FC 头的参数传给 AdamW。
- **OneCycleLR 调度器**：学习率先从一个较小值上升到 `max_lr=1e-3`（前 30% 的步数），然后余弦衰减到接近 0。这种"先升后降"的策略有助于跳出初始的不良局部最优。
- **`scheduler_per_batch=True`**：OneCycleLR 是按 batch 步进的（不是按 epoch），所以每个 batch 后都要调用 `scheduler.step()`。

#### 阶段一的作用

通过 8 个 epoch 的训练，FC 头从随机状态收敛到一个合理的状态。此时 FC 头已经能够基于 backbone 提取的特征做出比较准确的分类。接下来进入阶段二时，FC 头产生的梯度就不会再那么"疯狂"，backbone 的预训练权重也就不会被破坏了。

## 04.3 阶段二：全网络微调

#### 实现方式——解冻所有层

```python
def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True
```

一行代码，效果却很显著：现在所有约 2100 万个参数都参与训练了。backbone 中的卷积层权重也会根据 CIFAR-10 数据的特点进行微调。

#### 训练配置

```python
opt2 = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
sch2 = CosineAnnealingLR(opt2, T_max=30, eta_min=1e-6)
```

关键区别：

- **学习率是阶段一的 1/10**$(10^{-4}) vs (10^{-3}$)。小学习率确保对预训练权重的修改是"微调"而非"大改"，避免灾难性遗忘。
- **优化器接收所有参数**：`model.parameters()` 返回全部参数。
- **CosineAnnealingLR 调度器**：学习率按余弦曲线从 \(10^{-4}\) 缓慢衰减到 \(10^{-6}\)，`scheduler_per_batch=False`，即每个 epoch 结束后步进一次。
- **早停 patience=10**：如果验证集准确率连续 10 个 epoch 没有刷新最高记录，就停止训练。

## 04.4 两阶段流程总览

```css
时间轴 ──────────────────────────────────────────────────►

        ┌── 阶段一：预热 FC 头 ──┐┌── 阶段二：全网络微调 ──────────┐
        │ epoch 1 ~ 8            ││ epoch 9 ~ 38                   │
        │                        ││                                 │
        │ backbone: 冻结 ❄️       ││ backbone: 解冻 🔥               │
        │ FC 头:    训练 🔥       ││ FC 头:    训练 🔥               │
        │ 学习率:   1e-3（较大）  ││ 学习率:   1e-4（较小）          │
        │ 调度器:   OneCycleLR   ││ 调度器:   CosineAnnealingLR    │
        │ 目的: FC 头快速收敛     ││ 目的: 整体精细调整              │
        └────────────────────────┘└─────────────────────────────────┘
```

# 05 核心 API

## 05.1 单 epoch 训练流程

```python
def train_one_epoch(model, loader, optimizer, criterion, scheduler, device, ...):
    model.train()            # 切换到训练模式（启用 Dropout、BatchNorm 的训练行为）
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)  # 数据送入 GPU
        optimizer.zero_grad()     # 清除上一步的梯度
        out  = model(imgs)        # 前向传播
        loss = criterion(out, labels)  # 计算损失
        loss.backward()           # 反向传播，计算梯度
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()          # 更新参数
        if scheduler_per_batch:
            scheduler.step()      # 更新学习率
```

逐步解释关键 API：

**`model.train()` 与 `model.eval()`**

这两个方法切换模型的运行模式。区别在于：

- **Dropout 层**：`train()` 模式下随机丢弃神经元；`eval()` 模式下不丢弃，而是使用全部神经元（输出会自动乘以保留概率以保持期望值一致）。
- **BatchNorm 层**：`train()` 模式下使用当前 batch 的均值和方差做归一化，并更新滑动平均统计量；`eval()` 模式下使用训练期间积累的滑动平均统计量。

忘记切换模式是新手常犯的错误——验证时如果忘记调用 `model.eval()`，Dropout 仍在随机丢弃、BatchNorm 用的是当前 batch 的统计量，会导致评估结果不稳定。

**`optimizer.zero_grad()`**

PyTorch 的梯度默认是**累加**的（这在某些高级场景下有用），所以每次计算新梯度前必须先清零。

**`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`**

梯度裁剪。如果所有参数的梯度拼接成一个大向量，其 L2 范数超过了 `max_norm=1.0`，就按比例缩小所有梯度，使范数恰好等于 1.0。这可以防止"梯度爆炸"——某些 batch 可能恰好产生异常大的梯度，梯度裁剪能保护模型不被这些异常值破坏。

**`@torch.no_grad()`**

评估函数 `evaluate()` 和 `get_predictions()` 都带有这个装饰器。它的作用是告诉 PyTorch：这个函数内部不需要跟踪梯度计算图。好处是节省显存（不需要保存中间结果用于反向传播）和加速计算。

## 05.2 损失函数：CrossEntropyLoss + Label Smoothing

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

**CrossEntropyLoss** 是分类任务的标准损失函数。对于一个样本，它的计算逻辑是：

$$
\text{Loss} = -\sum_{i=1}^{C} y_i \cdot \log(\hat{y}_i)
$$
其中 \(C\) 是类别数，\(y_i\) 是真实标签（one-hot），$\hat{y}_i $是模型预测的概率（经过 softmax）。加上 `label_smoothing=0.1` 后，真实标签从硬编码的 `[0, 0, 1, 0, ..., 0]` 变成了 `[0.01, 0.01, 0.91, 0.01, ..., 0.01]`（具体公式是对正确类别分配 $1 - \alpha + \alpha/C$，其余类别分配 $\alpha/C$，其中 $\alpha=0.1$。这使得模型不会过度自信，提升了泛化能力

## 05.3 优化器：AdamW

```python
opt = optim.AdamW(params, lr=..., weight_decay=1e-4)
```

AdamW 是 Adam 优化器的改进版本。Adam 结合了动量（Momentum）和自适应学习率（RMSProp）的优点，而 AdamW 修正了 Adam 中权重衰减（Weight Decay）的实现方式——Adam 原版把权重衰减混在了梯度更新里，导致正则化效果不纯粹；AdamW 将两者解耦，使得权重衰减独立于梯度更新。`weight_decay=1e-4` 是 L2 正则化的强度，给模型权重加了一个"要保持较小值"的约束，防止过拟合

## 05.4 学习率调度器

代码中使用了两种不同的调度器，分别对应两个训练阶段：

**OneCycleLR（阶段一）**

```python
sch1 = OneCycleLR(opt1, max_lr=1e-3, steps_per_epoch=len(train_loader),
                  epochs=8, pct_start=0.3)
```

学习率变化轨迹：从很小 → 上升到 `max_lr` → 再下降到接近 0。整个周期中，前 30%（`pct_start=0.3`）用于上升，后 70% 用于下降。这种策略的直觉是：训练初期用小学习率"试探"，中间用大学习率快速学习，后期再用小学习率精细调整。

**CosineAnnealingLR（阶段二）**

```python
sch2 = CosineAnnealingLR(opt2, T_max=30, eta_min=1e-6)
```

学习率按余弦函数从初始值 \(10^{-4}\) 平滑衰减到 \(10^{-6}\)。`T_max=30` 表示一个完整余弦周期为 30 个 epoch。余弦衰减比线性衰减更平滑，前期衰减慢（保持较大学习率充分学习），后期衰减快（精细收敛）。

## 05.5 早停机制（Early Stopping）

```python
if vl_acc > best_acc:
    best_acc = vl_acc
    best_wts = copy.deepcopy(model.state_dict())  # 保存当前最优权重
    no_improve = 0
else:
    no_improve += 1

if no_improve >= patience:
    print("早停触发")
    break
```

核心逻辑：

1. 每个 epoch 后评估验证集准确率。
2. 如果创了新高，就用 `copy.deepcopy(model.state_dict())` 深拷贝一份当前模型的全部参数。
3. 如果连续 `patience` 个 epoch 都没有刷新最高记录，就认为模型已经开始过拟合，终止训练。
4. 训练结束后，加载的是**历史最优权重**（`model.load_state_dict(best_wts)`），而不是最后一个 epoch 的权重。

`model.state_dict()` 返回一个有序字典，包含模型所有参数的名称和值（张量），例如 `{'conv1.weight': tensor(...), 'bn1.weight': tensor(...), ...}`。`copy.deepcopy()` 是为了做完整的深拷贝，否则字典中的张量仍然是对模型参数的引用，后续训练会修改它们

# 06 完整代码

```python
"""
================================================================================
  迁移学习策略二：Full Fine-tuning（全网络微调）
================================================================================
  核心思想：
    加载预训练权重后，解冻所有层，使用统一的小学习率对整个网络端到端训练。
    预训练权重作为优质初始值，所有层的参数都会随训练更新。

    推荐两阶段流程：
      阶段一（预热）：先只训练 FC 头几个 epoch，让随机初始化的 FC 先收敛，
                     避免初期大梯度破坏 backbone 的预训练权重。
      阶段二（微调）：解冻全网络，用小学习率做端到端微调。

  适用场景：
    - 目标数据集规模中等或较大（> 10000 张）
    - 目标域与 ImageNet 有一定差异，仅特征提取效果不足

  运行方式：
    python strategy2_full_finetune.py

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
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader, random_split

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "SimHei",              # Windows 常见
    "Microsoft YaHei",     # Windows 常见
    "PingFang SC",         # macOS 常见
    "Noto Sans CJK SC",    # Linux/跨平台常见
]
plt.rcParams["axes.unicode_minus"] = False  # 避免负号显示成方块


# ============================================================
#  全局配置
# ============================================================
CFG = {
    "strategy_name":   "Full Fine-tuning",
    "data_dir":        "./data",
    "save_dir":        "./results/strategy2_full_finetune",
    "backbone":        "resnet34",
    "num_classes":     10,
    "batch_size":      128,
    "num_workers":     2,
    "val_split":       0.1,
    "seed":            42,
    "device":          "cuda" if torch.cuda.is_available() else "cpu",
    # 阶段一：预热 FC 头
    "warmup_epochs":   12,
    "warmup_lr":       1e-3,
    # 阶段二：全网络微调
    "finetune_epochs": 40,
    "finetune_lr":     1e-4,       # 必须远小于从头训练的学习率，防止灾难性遗忘
    "weight_decay":    1e-4,
    "label_smoothing": 0.1,
    "patience":        10,
}

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
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
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(degrees=15),
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

    full_val = torchvision.datasets.CIFAR10(
        root=cfg["data_dir"], train=True, download=False, transform=val_tf)
    val_dataset = torch.utils.data.Subset(full_val, val_sub.indices)

    test_dataset = torchvision.datasets.CIFAR10(
        root=cfg["data_dir"], train=False, download=True, transform=val_tf)

    kw = dict(num_workers=cfg["num_workers"], pin_memory=True)
    train_loader = DataLoader(train_sub,    batch_size=cfg["batch_size"],   shuffle=True,  **kw)
    val_loader   = DataLoader(val_dataset,  batch_size=cfg["batch_size"]*2, shuffle=False, **kw)
    test_loader  = DataLoader(test_dataset, batch_size=cfg["batch_size"]*2, shuffle=False, **kw)

    print(f"[数据] 训练: {n_train}  验证: {n_val}  测试: {len(test_dataset)}")
    return train_loader, val_loader, test_loader


# ============================================================
#  2. 模型构建
# ============================================================
def build_model(cfg):
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    model.maxpool = nn.Identity()
    in_features = model.fc.in_features   # ResNet50: 2048
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.15),
        nn.Linear(512, cfg["num_classes"])
    )
    return model


# ============================================================
#  3. 策略核心：两阶段训练配置
# ============================================================
def freeze_backbone(model):
    """冻结所有层，只开放 FC 头和改造后的conv1（阶段一：预热）"""
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    # 同时解冻被替换的 conv1（它是随机初始化的，需要训练）
    for param in model.conv1.parameters():
        param.requires_grad = True


def unfreeze_all(model):
    """解冻全部层（阶段二：全网络微调）"""
    for param in model.parameters():
        param.requires_grad = True


def _print_param_status(model, stage_name):
    print(f"\n[{stage_name}] 参数状态：")
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total:,}  可训练: {trainable:,}  ({100*trainable/total:.1f}%)")
    for name, module in model.named_children():
        params = list(module.parameters())
        if not params:
            continue
        t = sum(p.numel() for p in params)
        tr = sum(p.numel() for p in params if p.requires_grad)
        status = "✓ 训练" if tr == t else ("✗ 冻结" if tr == 0 else "部分")
        print(f"  {name:<10} {status}")


# ============================================================
#  4. 训练与评估
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, scheduler, device, scheduler_per_batch=True):
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
        if scheduler_per_batch and scheduler is not None:
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


def run_stage(model, train_loader, val_loader, optimizer, scheduler, criterion,
              cfg, n_epochs, stage_name, patience, scheduler_per_batch=True):
    """通用训练阶段循环（带早停和最优模型保存）"""
    device     = cfg["device"]
    best_acc   = 0.0
    best_wts   = copy.deepcopy(model.state_dict())
    no_improve = 0
    history    = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    print(f"\n{'─'*60}")
    print(f"  [{stage_name}]  共 {n_epochs} epochs")
    print(f"{'─'*60}")

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scheduler, device, scheduler_per_batch)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        if not scheduler_per_batch and scheduler is not None:
            scheduler.step()
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
            flag = "  ← best"
            no_improve = 0
        else:
            no_improve += 1

        print(f"  Epoch [{epoch:3d}/{n_epochs}] "
              f"train={tr_acc*100:.2f}%  val={vl_acc*100:.2f}%  "
              f"loss={vl_loss:.4f}  ({elapsed:.1f}s){flag}")

        if no_improve >= patience:
            print(f"  早停触发（连续 {patience} epoch 无提升）")
            break

    model.load_state_dict(best_wts)
    print(f"  [{stage_name}] 最优验证集准确率: {best_acc*100:.2f}%")
    return model, history, best_acc


def run_training(model, train_loader, val_loader, cfg):
    """
    全网络微调的两阶段训练流程：
      阶段一：冻结 backbone，只训练 FC 头（预热）
      阶段二：解冻全网络，小学习率全量微调
    """
    device    = cfg["device"]
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    print(f"\n{'='*60}")
    print(f"  策略二：{cfg['strategy_name']}")
    print(f"  设备: {device.upper()}")
    print(f"{'='*60}")

    # ── 阶段一：预热 FC 头 ──
    freeze_backbone(model)
    _print_param_status(model, "阶段一·预热")

    warmup_params = [p for p in model.parameters() if p.requires_grad]
    opt1 = optim.AdamW(warmup_params, lr=cfg["warmup_lr"], weight_decay=cfg["weight_decay"])
    sch1 = OneCycleLR(
        opt1, max_lr=cfg["warmup_lr"],
        steps_per_epoch=len(train_loader), epochs=cfg["warmup_epochs"],
        pct_start=0.3)
    model, hist1, _ = run_stage(
        model, train_loader, val_loader, opt1, sch1, criterion,
        cfg, cfg["warmup_epochs"], "阶段一·FC头预热", patience=5, scheduler_per_batch=True)

    # ── 阶段二：全网络微调 ──
    unfreeze_all(model)
    _print_param_status(model, "阶段二·全量微调")

    # 注意：全网络微调学习率必须比预热阶段小 10x，防止灾难性遗忘
    opt2 = optim.AdamW(model.parameters(), lr=cfg["finetune_lr"], weight_decay=cfg["weight_decay"])
    sch2 = CosineAnnealingLR(opt2, T_max=cfg["finetune_epochs"], eta_min=1e-6)
    model, hist2, _ = run_stage(
        model, train_loader, val_loader, opt2, sch2, criterion,
        cfg, cfg["finetune_epochs"], "阶段二·全网络微调", patience=cfg["patience"],
        scheduler_per_batch=False)

    # 合并两阶段历史（在可视化时加分隔线）
    history = {
        k: hist1[k] + hist2[k]
        for k in ("train_loss", "train_acc", "val_loss", "val_acc", "lr")
    }
    history["warmup_end_epoch"] = len(hist1["train_loss"])
    return model, history


# ============================================================
#  5. 可视化
# ============================================================
def plot_training_curves(history, cfg):
    """绘制训练曲线，并标注两阶段分界线"""
    epochs   = range(1, len(history["train_loss"]) + 1)
    sep      = history.get("warmup_end_epoch", None)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"训练曲线 — {cfg['strategy_name']}", fontsize=14, fontweight="bold")

    def add_sep(ax):
        if sep:
            ax.axvline(x=sep, color="#FF9800", linestyle="--", linewidth=1.5,
                       alpha=0.8, label=f"阶段分界 (epoch {sep})")

    # Loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train Loss", color="#2196F3", linewidth=2)
    ax.plot(epochs, history["val_loss"],   label="Val Loss",   color="#FF5722", linewidth=2)
    add_sep(ax)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss 曲线")
    ax.legend(); ax.grid(alpha=0.3)

    # Accuracy
    ax = axes[1]
    tr_acc = [a * 100 for a in history["train_acc"]]
    vl_acc = [a * 100 for a in history["val_acc"]]
    ax.plot(epochs, tr_acc, label="Train Acc", color="#2196F3", linewidth=2)
    ax.plot(epochs, vl_acc, label="Val Acc",   color="#FF5722", linewidth=2)
    best_ep  = int(np.argmax(history["val_acc"])) + 1
    ax.axvline(x=best_ep, color="gray", linestyle=":", alpha=0.7, label=f"Best epoch={best_ep}")
    ax.scatter([best_ep], [max(vl_acc)], color="#FF5722", zorder=5, s=80)
    add_sep(ax)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)"); ax.set_title("Accuracy 曲线")
    ax.legend(); ax.grid(alpha=0.3)

    # LR
    ax = axes[2]
    ax.plot(epochs, history["lr"], color="#4CAF50", linewidth=2)
    add_sep(ax)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate"); ax.set_title("学习率变化")
    ax.legend(); ax.grid(alpha=0.3)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    plt.tight_layout()
    path = os.path.join(cfg["save_dir"], "01_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[可视化] 训练曲线 → {path}")


def plot_confusion_matrix(y_true, y_pred, cfg):
    cm      = confusion_matrix(y_true, y_pred)
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
    cm  = confusion_matrix(y_true, y_pred)
    acc = cm.diagonal() / cm.sum(axis=1) * 100
    colors = plt.cm.RdYlGn(acc / 100)
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(CIFAR10_CLASSES, acc, color=colors, edgecolor="white", linewidth=0.8)
    for bar, a in zip(bars, acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{a:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(y=np.mean(acc), color="#2196F3", linestyle="--", linewidth=2,
               label=f"平均: {np.mean(acc):.1f}%")
    ax.set_ylim(0, 110); ax.set_xlabel("类别"); ax.set_ylabel("准确率 (%)")
    ax.set_title(f"各类别准确率 — {cfg['strategy_name']}", fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(cfg["save_dir"], "03_per_class_accuracy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[可视化] 各类别准确率 → {path}")


def plot_sample_predictions(model, test_loader, cfg, n_samples=40):
    device = cfg["device"]
    model.eval()
    images_list, labels_list, preds_list = [], [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu()
            images_list.append(imgs.cpu()); labels_list.append(labels); preds_list.append(preds)
            if sum(len(x) for x in labels_list) >= n_samples:
                break
    images = torch.cat(images_list)[:n_samples]
    labels = torch.cat(labels_list)[:n_samples]
    preds  = torch.cat(preds_list)[:n_samples]
    mean   = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std    = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    images = (images * std + mean).clamp(0, 1)
    cols = 8; rows = n_samples // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 2.2))
    fig.suptitle(f"预测样本展示 — {cfg['strategy_name']}\n绿框=正确  红框=错误",
                 fontsize=12, fontweight="bold")
    for idx, ax in enumerate(axes.flat):
        ax.imshow(images[idx].permute(1, 2, 0).numpy()); ax.axis("off")
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
    ax.set_xlabel("错误次数"); ax.grid(axis="x", alpha=0.3)
    ax.set_title(f"Top-{top_n} 易混淆类别对 — {cfg['strategy_name']}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(cfg["save_dir"], "05_top_errors.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[可视化] 易混淆类别 → {path}")


def plot_two_stage_comparison(history, cfg):
    """
    策略二特有的可视化：对比两阶段（预热 vs 全量微调）的 val_acc 增长曲线
    """
    sep    = history.get("warmup_end_epoch", 0)
    epochs = list(range(1, len(history["val_acc"]) + 1))
    vl_acc = [a * 100 for a in history["val_acc"]]

    fig, ax = plt.subplots(figsize=(12, 5))
    # 背景分区
    ax.axvspan(0, sep + 0.5, alpha=0.08, color="#2196F3", label="阶段一：预热 FC 头")
    ax.axvspan(sep + 0.5, epochs[-1], alpha=0.08, color="#FF5722", label="阶段二：全网络微调")
    ax.plot(epochs, vl_acc, color="#333", linewidth=2.5, label="Val Accuracy")
    ax.axvline(x=sep + 0.5, color="#FF9800", linewidth=2, linestyle="--", label=f"阶段切换 (epoch {sep})")
    ax.scatter([int(np.argmax(vl_acc)) + 1], [max(vl_acc)],
               color="#FF5722", s=100, zorder=5, label=f"最优: {max(vl_acc):.2f}%")
    ax.set_xlabel("Epoch", fontsize=12); ax.set_ylabel("Val Accuracy (%)", fontsize=12)
    ax.set_title("两阶段训练对比：预热 → 全网络微调", fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(cfg["save_dir"], "06_two_stage_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[可视化] 两阶段对比 → {path}")


def save_summary(history, y_true, y_pred, cfg, test_acc):
    report = classification_report(y_true, y_pred, target_names=CIFAR10_CLASSES)
    path = os.path.join(cfg["save_dir"], "summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\n  训练策略: {cfg['strategy_name']}\n")
        f.write(f"  预热 Epochs: {cfg['warmup_epochs']}\n")
        f.write(f"  微调 Epochs: {cfg['finetune_epochs']}\n")
        f.write(f"  总 Epoch 数: {len(history['train_loss'])}\n")
        f.write(f"  最优验证集准确率: {max(history['val_acc'])*100:.2f}%\n")
        f.write(f"  测试集准确率:     {test_acc*100:.2f}%\n")
        f.write(f"{'='*60}\n\n分类报告:\n{report}")
    print(f"[摘要] 训练报告 → {path}")


# ============================================================
#  主程序
# ============================================================
def main():
    cfg    = CFG
    device = cfg["device"]
    print(f"\n{'='*60}")
    print(f"  策略二：{cfg['strategy_name']}")
    print(f"  设备: {device.upper()}" +
          (f"  ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    print(f"{'='*60}")

    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    model, history = run_training(model, train_loader, val_loader, cfg)

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n[测试集] Loss={test_loss:.4f}  Accuracy={test_acc*100:.2f}%")

    y_pred, y_true, y_prob = get_predictions(model, test_loader, device)

    print("\n[可视化] 生成图表中...")
    plot_training_curves(history, cfg)
    plot_confusion_matrix(y_true, y_pred, cfg)
    plot_per_class_accuracy(y_true, y_pred, cfg)
    plot_sample_predictions(model, test_loader, cfg)
    plot_top_errors(y_true, y_pred, y_prob, cfg)
    plot_two_stage_comparison(history, cfg)
    save_summary(history, y_true, y_pred, cfg, test_acc)

    print(f"\n所有结果已保存至: {cfg['save_dir']}/")
    print("完成！")


if __name__ == "__main__":
    main()
```

