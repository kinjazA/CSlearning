# 01 AlexNet

> * AlexNet 是一个 8 层深度卷积神经网络（5 个卷积层 + 3 个全连接层），总参数约 6000 万。它在 2012 年 ImageNet 竞赛中以 15.3% 的 Top-5 错误率夺冠，远超第二名的 26.2%，标志着深度学习在图像识别领域的首次大胜。
> * ==结构包括大核卷积开头、最大池化降维、ReLU 激活、全连接分类头，以及创新的 Dropout 和数据增强==
> * 它解决了梯度消失（用 ReLU）、过拟合（用 Dropout 和增强）、计算瓶颈（GPU 并行），直接引发了深度学习热潮。如果没有 AlexNet，ResNet 等后续模型可能延迟几年出现——它像“蒸汽机”一样，开启了 AI 工业革命
> * ==AlexNet 的创新不是单纯堆层，而是系统性技巧组合：大核开头 + 小核堆叠、ReLU、Dropout、数据增强、GPU 并行。通道从 96 开始翻倍，池化每 2-3 层一次==

**目前还在借鉴的思想**

- ReLU + Dropout + 数据增强已成为所有 CNN 训练的“铁三角”
- “大核开头 + 小核堆叠”的分层设计影响了 VGG 和 ResNet

## 01.1 数据增强方法

> * 数据增强是 AlexNet 能用 120 万张 ImageNet 数据“虚拟”扩充到上亿张的关键。它不是简单翻转，而是系统性模拟现实世界中的“变化”，让模型学到不变的本质特征（比如猫不管旋转、裁剪、亮度变化都认得出）
> * 真实世界图片永远有噪声：光照不同、角度不同、位置偏移。如果只用原始数据，模型会“死记硬背”具体像素，导致严重过拟合（训练集 99%，测试集 70%）。数据增强通过随机变换，迫使模型学习“鲁棒特征”，相当于免费扩充数据集 10-100 倍，精度提升 5-15%。AlexNet 论文里明确说：没有增强，错误率会高 10%以上
> * 数据增强的核心思想是：**在不收集新数据的前提下，通过对已有数据做合理的变换，人为制造出更多的训练样本**，让网络见识到同一张图片在不同"条件"下的样子，从而学到更鲁棒（Robust）的特征
> * 关键词是"合理的变换"。什么叫合理？就是变换之后，图片的**语义标签不变**。比如一张猫的图片水平翻转之后，它还是猫；亮度稍微调暗之后，它还是猫。但是如果你把猫的图片旋转 180 度，对于某些任务（比如数字识别，6 和 9 就不同了）就是不合理的变换了。所以数据增强的设计必须结合具体任务来考量

一条典型的分类增强管线，按“物理世界”思路是：

1. **几何类**：裁剪 / 翻转 / 缩放 / 旋转
    → 模拟：拍照角度、构图、物体位置变化
2. **颜色类**：亮度/对比度/饱和度/色相
    → 模拟：光照、相机曝光、色偏
3. **转成模型能吃的数**：把图片变成 Tensor（以及 dtype/range 对齐）
4. **Normalize**：标准化让训练更稳

官方也强调：Tensor 图像的数值范围由 dtype 隐式决定：**float 期望 [0,1]，uint8 期望 [0,255]**，并推荐用 `ToDtype` 统一 dtype 和范围。

### 1.torchvision.transforms

```python
import torchvision.transforms as transforms

# ===========================================================
# 训练集的数据增强 pipeline
# transforms.Compose 将多个变换串联起来，按顺序依次执行。
# 每次从 DataLoader 取一张图时，都会重新随机执行这些变换，
# 所以同一张图在不同 epoch 会产生不同的变体。
# ===========================================================
transform_train = transforms.Compose([

    # -----------------------------------------------------------
    # Step 1: RandomCrop —— 随机裁剪
    # -----------------------------------------------------------
    # 作用：先在图像四周填充 padding 个像素（默认用0填充，即黑边），
    #       然后从填充后的图像中随机裁剪出 size × size 的区域。
    #
    # 参数详解：
    #   size=32       : 裁剪出来的目标尺寸，和原图一样大（32×32）
    #   padding=4     : 四周各填充4像素，图像变成 40×40，再裁出32×32
    #                   这意味着图像内容最多可以偏移4个像素，
    #                   模拟了轻微的位移不变性。
    #   padding_mode  : 默认是 'constant'（填0，黑边）
    #                   也可以用 'reflect'（镜像填充，更自然）
    #                   或 'edge'（边缘像素复制填充）
    #
    # 为什么用 padding=4？
    #   CIFAR-10 图像只有32像素，padding=4 相当于允许12.5%的偏移，
    #   是一个经过实验验证的合理比例。padding太大会引入太多黑边，
    #   padding太小则增强效果不明显。
    #
    # 直觉理解：
    #   想象把图像贴在一个稍大的黑色画布上，然后随机"截取"原图大小的窗口。
    #   这让模型学会：同一个物体出现在图像的不同位置都应该被正确识别。
    # -----------------------------------------------------------
    transforms.RandomCrop(size=32, padding=4),

    # -----------------------------------------------------------
    # Step 2: RandomHorizontalFlip —— 随机水平翻转
    # -----------------------------------------------------------
    # 作用：以 p 的概率对图像做左右镜像翻转。
    #
    # 参数详解：
    #   p=0.5 : 翻转概率，默认0.5，即每张图有50%的概率被翻转。
    #           这是最常用的值，使翻转/不翻转各占一半，分布均衡。
    #
    # 为什么不用 RandomVerticalFlip？
    #   自然图像有重力方向的语义约束，猫、狗、汽车等物体倒置后
    #   不符合真实分布，会引入噪声样本，通常不做垂直翻转。
    #   特殊场景除外：卫星图像、显合理的增强手段。
    #
    # 直觉理解：
    #   一只向左跑的猫和一只向右跑的猫，本质上是同一类对象。
    #   水平翻转让模型学到左右对称不变性。
    # -----------------------------------------------------------
    transforms.RandomHorizontalFlip(p=0.5),

    # -----------------------------------------------------------
    # Step 3: ToTensor —— 转为 Tensor
    # -----------------------------------------------------------
    # 作用：将 PIL Image 或 numpy array（H × W × C，值域[0,255]）
    #       转换为 PyTorch Tensor（C × H × W，值域[0.0,1.0]）。
    #
    # 注意事项（易错点！）：
    #   1. 维度顺序变了：HWC → CHW。PyTorch 的卷积层期望输入是 (N,C,H,W)。
    #   2. 值域自动缩放：[0,255] → [0.0,1.0]，相当于除以255。
    #   3. 必须在 Normalize 之前调用，因为 Normalize 要求输入是 Tensor。
    #   4. 如果输入已经是 float 类型的 numpy array（比如你手动加载的图），
    #      ToTensor 不会做值域缩放，只做维度转换。
    # -----------------------------------------------------------
    transforms.ToTensor(),

    # -----------------------------------------------------------
    # Step 4: Normalize —— 标准化
    # -----------------------------------------------------------
    # 作用：对每个通道做 z-score 标准化：
    #       output[c] = (input[c] - mean[c]) / std[c]
    #
    # 参数详解：
    #   mean=(0.4914, 0.4822, 0.4465) : CIFAR-10 训练集 R/G/B 三通道的均值
    #   std=(0.2023, 0.1994, 0.2010)  : CIFAR-10 训练集 R/G/B 三通道的标准差
    #
    # 为什么要标准化？
    #   1. 梯度数值稳定性：未标准化时不同通道值域差异大，梯度更新不均匀。
    #   2. 加速收敛：标准化后输入分布接近零均值单位方差，符合权重初始化的假设。
    #   3. 防止某个通道主导学习：若 R 通道值普遍偏大，模型可能过度依赖 R 通道。
    #
    # 这里的均值和标准差是怎么来的？
    #   是提前对 CIFAR-10 全部50000张训练图像统计出来的，是固定的经验值。
    #   计算方法如下（仅供理解，实际直接用这组数就好）：
    #
    #     import torchvision
    #     import torch
    #     dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
    #                  transform=transforms.ToTensor(), download=True)
    #     loader = torch.utils.data.DataLoader(dataset, batch_size=50000)
    #     data = next(iter(loader))[0]  # shape: (50000, 3, 32, 32)
    #     mean = data.mean(dim=[0,2,3])  # 对 N,H,W 维度求均值，保留 C 维度
    #     std  = data.std(dim=[0,2,3])
    #     # mean ≈ [0.4914, 0.4822, 0.4465]
    #     # std  ≈ [0.2023, 0.1994, 0.2010]
    #
    # 极其重要的易错点：
    #   测试集的 Normalize 必须使用与训练集完全相同的 mean 和 std！
    #   不能对测试集重新统计 mean/std，因为在真实场景中你看不到测试集，
    #   标准化参数必须只从训练集得到。
    # -----------------------------------------------------------
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    ),
])


# ===========================================================
# 测试集的 pipeline：不做随机增强，只做确定性预处理
# ===========================================================
# 为什么测试集不做数据增强？
#   1. 测试集的目的是评估模型的真实泛化能力，必须使用确定性变换，
#      否则同一张图每次评估结果不同，无法复现。
#   2. 随机裁剪/翻转会改变图像内容，可能导致本来正确的预测变错。
# ===========================================================
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),   # 必须与训练集一致
        std=(0.2023, 0.1994, 0.2010)
    ),
])
```

### 2. 可视化验证

```python
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np

# ---- 加载一张原始图像（不做任何增强）----
raw_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transforms.ToTensor()   # 只转Tensor，不做增强，用于展示原图
)

# CIFAR-10 的类别名
classes = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']

# 取一张图像（选第 0 张）
original_img_tensor, label = raw_dataset[0]
# original_img_tensor shape: (3, 32, 32)，值域 [0,1]

# 将 Tensor 转回 numpy 用于显示
# 注意：matplotlib 需要 HWC 格式，值域 [0,1] 的 float 或 [0,255] 的 uint8
def tensor_to_numpy(t):
    """将 (C,H,W) 的Tensor转为 (H,W,C) 的numpy，用于imshow"""
    return t.permute(1, 2, 0).numpy()  # CHW → HWC

# ---- 定义各种单独的增强变换，用于对比展示 ----

# 注意：这里每个 transform 都要先从 PIL 图像开始，
# 所以要先把 Tensor 转回 PIL Image
to_pil    = transforms.ToPILImage()
to_tensor = transforms.ToTensor()

# 各种增强方式
augmentations = {
    "Original":          transforms.Compose([to_tensor]),
    "RandomCrop(pad=4)": transforms.Compose([
                             transforms.RandomCrop(32, padding=4),
                             to_tensor]),
    "HorizontalFlip":    transforms.Compose([
                             transforms.RandomHorizontalFlip(p=1.0),  # p=1强制翻转
                             to_tensor]),
    "ColorJitter":       transforms.Compose([
                             transforms.ColorJitter(
                                 brightness=0.4,   # 亮度在 [1-0.4, 1+0.4] 之间随机变化
                                 contrast=0.4,     # 对比度同上
                                 saturation=0.4,   # 饱和度同上
                                 hue=0.1           # 色调在 [-0.1, 0.1] 之间随机变化
                             ),
                             to_tensor]),
    "RandomRotation":    transforms.Compose([
                             transforms.RandomRotation(degrees=15),  # 随机旋转±15度
                             to_tensor]),
    "Full overlay":          transforms.Compose([
                             transforms.RandomCrop(32, padding=4),
                             transforms.RandomHorizontalFlip(p=0.5),
                             transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                             to_tensor]),
}

# ---- 可视化 ----
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

# 先把 Tensor 转回 PIL Image（大部分 transform 要求 PIL 输入）
pil_img = to_pil(original_img_tensor)

for idx, (name, aug) in enumerate(augmentations.items()):
    # 每次对同一张 PIL 图像应用不同的增强
    augmented_tensor = aug(pil_img)
    axes[idx].imshow(tensor_to_numpy(augmented_tensor))
    axes[idx].set_title(name, fontsize=12)
    axes[idx].axis('off')

plt.suptitle(f'original label: {classes[label]}', fontsize=14)
plt.tight_layout()
plt.savefig('augmentation_visualization.png', dpi=150)
plt.show()
print("可视化图已保存为 augmentation_visualization.png")
```

![image-20260225203628295](F:\note\deep_learning\pytorch_learning\day08_AlexNet.assets\image-20260225203628295.png)

## 01.2 进阶数据增强 API 

### 1. ColorJitter —— 颜色扰动

```python
transforms.ColorJitter(
    brightness=0.4,   # 亮度因子在 [max(0, 1-0.4), 1+0.4] = [0.6, 1.4] 之间均匀采样
                      # 如果传入 (min, max) 元组则直接指定范围
    contrast=0.4,     # 对比度：增强区分明暗的能力
    saturation=0.4,   # 饱和度：控制颜色鲜艳程度，0为灰度图
    hue=0.1           # 色调偏移量，范围 [-0.5, 0.5]，0.1 是很轻微的偏移，注意：hue 偏移太大会使颜色失真，超出自然图像分布
)
```

**适用场景**：户外场景（光照变化大）、医学图像（仪器颜色差异）、工业检测（环境光变化）。

**不适用场景**：颜色本身是判断依据的任务，比如红绿灯检测、水果成熟度判断（颜色改变会改变标签语义）

### 2. RandomRotation —— 随机旋转

```python
transforms.RandomRotation(
    degrees=15,          # 旋转角度范围 [-15°, +15°]，也可以传 (min, max) 元组
    interpolation=transforms.InterpolationMode.BILINEAR,  # 插值方式，双线性插值更平滑
    expand=False,        # False：旋转后裁剪到原始尺寸（可能有角落空白）
                         # True：扩展画布以包含完整旋转图像（尺寸会变化）
    fill=0               # 旋转后空白区域填充的像素值，0 为黑色
)
```

**适用场景**：医学图像（X光、CT，器官可能有不同角度）、遥感图像（无方向性）、通用分类任务（小角度旋转）。

**不适用场景**：文字识别（倒置的字母语义不同）、目标检测（旋转后 bounding box 会不准）

### 3. RandomGrayscale —— 随机转灰度

```python
transforms.RandomGrayscale(
    p=0.1    # 以 10% 的概率将彩色图转为灰度图（但保持3个通道，三个通道值相同）
)
```

**作用**：让模型不过度依赖颜色信息，增强对纹理和形状的学习能力。SimCLR 等对比学习方法中经常用到

### 4. RandomErasing —— 随机遮挡（Cutout）

```python
# 注意：RandomErasing 必须在 ToTensor() 之后使用，因为它操作的是 Tensor
transforms.RandomErasing(
    p=0.5,              # 执行遮挡的概率
    scale=(0.02, 0.33), # 遮挡区域面积占图像总面积的比例范围
    ratio=(0.3, 3.3),   # 遮挡区域的宽高比范围
    value=0,            # 遮挡区域填充值，0为黑色；也可以用 'random' 填充随机噪声
    inplace=False
)
```

**直觉**：模拟物体被部分遮挡的情况，让模型学会用局部特征做判断，而不是依赖整体外观

**注意**：它必须放在 `ToTensor()` 之后，因为它操作的是 Tensor，不是 PIL Image

### 5. AutoAugment —— 自动增强策略

```python
# AutoAugment 是 Google 用强化学习搜索出的最优增强策略组合
# 内置了多种针对不同数据集优化的策略
transforms.AutoAugment(
    policy=transforms.AutoAugmentPolicy.CIFAR10  # 专门针对CIFAR-10优化的策略
    # 其他选项：IMAGENET（针对ImageNet）、SVHN（针对街景数字）
)
```

# 02 CIFAR10_AlexNetimport torch

```python
import torch.nn as nn

import torch.nn.functional as F



class AlexNet_CIFAR10(nn.Module):

    """

    针对 CIFAR-10（输入尺寸 32×32×3）适配的 AlexNet 实现。


    【关于 nn.Module】

    所有自定义的神经网络模型都必须继承自 nn.Module。

    nn.Module 提供了很多底层能力，比如：

      - 自动追踪模型内所有的可学习参数（weights 和 bias）

      - 提供 .to(device) 方法将整个模型移到 GPU

      - 提供 .train() / .eval() 方法切换训练/评估模式

      - 提供 .parameters() 方法返回所有参数供优化器使用

      - 提供 .state_dict() / .load_state_dict() 方法保存和加载模型权重


    继承 nn.Module 之后，你必须实现两个方法：

      - init：定义模型用到的所有层（卷积、全连接、激活等）

      - forward：定义数据从输入到输出的流动路径（前向传播逻辑）


    【为什么层要在 init 里定义而不是 forward 里？】

    因为只有在 init 里通过 self.xxx = nn.Xxx(...) 的形式赋值给实例属性，

    nn.Module 才能自动追踪到这些层的参数（weights、bias）。

    如果在 forward 里临时创建层，参数就不会被追踪，优化器也无法更新它们。

    """


    def init(self, num_classes=10):

        """

        init 方法：定义模型所需的所有层。


        参数：

            num_classes (int): 分类的类别数，CIFAR-10 是 10 类。

                               保留这个参数使模型更通用，

                               换个数据集时只需修改这里，不用改网络结构。

        """


        # 【必须调用父类的 init】

        # super().init() 调用 nn.Module 的初始化方法，

        # 完成一些内部状态的注册（参数字典、子模块字典等）。

        # 如果不调用这行，后续所有 nn.Module 的功能都会失效，报错。

        # 这是继承 nn.Module 的固定写法，每次都要有。

        super(AlexNet_CIFAR10, self).init()


        # ==============

        # 卷积层 1（Conv1）

        # ==============

        # 输入：(batch_size, 3, 32, 32)

        # 输出：(batch_size, 64, 32, 32)

        #

        # nn.Conv2d 参数详解：

        #   in_channels=3   : 输入的通道数，RGB图像是3通道

        #   out_channels=64 : 卷积核的数量，即输出特征图的通道数

        #                     每个卷积核学习一种特征（边缘、纹理等）

        #   kernel_size=3   : 卷积核的空间尺寸，3 即 3×3 的滑动窗口

        #   stride=1        : 卷积核每次滑动的步长，1 意味着逐像素滑动

        #   padding=1       : 在输入特征图的四周各补 1 圈像素（默认补0）

        #                     padding=1 配合 kernel=3 可保持空间尺寸不变

        #

        # 为什么不用原版 AlexNet 的 kernel=11, stride=4？

        #   原版针对 224×224 输入设计，第一层用大卷积核快速下采样是合理的。

        #   但 CIFAR-10 的输入只有 32×32，

        #   kernel=11, stride=4 会直接把尺寸压到 floor((32-11)/4)+1 = 6，

        #   后续几层池化后尺寸会变成负数，网络根本无法运行。

        self.conv1 = nn.Conv2d(

            in_channels=3,

            out_channels=64,

            kernel_size=3,

            stride=1,

            padding=1

        )


        # ==============

        # 卷积层 2（Conv2）

        # ==============

        # 输入：(batch_size, 64, 16, 16)  ← 经过 pool1 下采样后的尺寸

        # 输出：(batch_size, 192, 16, 16)

        #

        # 注意 in_channels=64：必须和上一层的 out_channels 保持一致！

        # 理解方式：上一层输出了64张特征图，这一层的每个卷积核

        # 都要在这64张特征图上滑动，所以输入通道数是64。

        self.conv2 = nn.Conv2d(

            in_channels=64,

            out_channels=192,

            kernel_size=3,

            stride=1,

            padding=1

        )


        # ==============

        # 卷积层 3（Conv3）

        # ==============

        # 输入：(batch_size, 192, 8, 8)  ← 经过 pool2 下采样后的尺寸

        # 输出：(batch_size, 384, 8, 8)

        #

        # 这一层通道数从192扩展到384，让网络学习更丰富的高层特征。

        # 空间尺寸：floor((8 + 21 - 3) / 1) + 1 = 8，不变。

        self.conv3 = nn.Conv2d(

            in_channels=192,

            out_channels=384,

            kernel_size=3,

            stride=1,

            padding=1

        )


        # ==============

        # 卷积层 4（Conv4）

        # ==============

        # 输入：(batch_size, 384, 8, 8)

        # 输出：(batch_size, 256, 8, 8)

        #

        # 注意：这里通道数从384减少到256，这是有意为之的。

        # 在深层网络中，适当收缩通道数可以减少参数量，

        # 同时也是在对特征进行"提炼"——把384个特征图压缩到256个更重要的特征表示。

        self.conv4 = nn.Conv2d(

            in_channels=384,

            out_channels=256,

            kernel_size=3,

            stride=1,

            padding=1

        )


        # ==============

        # 卷积层 5（Conv5）

        # ==============

        # 输入：(batch_size, 256, 8, 8)

        # 输出：(batch_size, 256, 8, 8)

        #

        # 最后一个征提取。

        # 之后会经过 pool3 下采样到 4×4，然后展平送入全连接层。

        self.conv5 = nn.Conv2d(

            in_channels=256,

            out_channels=256,

            kernel_size=3,

            stride=1,

            padding=1

        )


        # ==============

        # 池化层（MaxPool2d）

        # ==============

        # 三个卷积块各跟一个最大池化层，这里只需定义一个，

        # 因为三次池化的参数完全相同，可以复用同一个层对象。

        #

        # nn.MaxPool2d 参数详解：

        #   kernel_size=2 : 池化窗口大小，2×2 的区域取最大值

        #   stride=2      : 步长等于窗口大小，意味着不重叠池化

        #                   每次下采样后空间尺寸减半：32→16→8→4

        #

        # 为什么用最大池化而不是平均池化？

        #   最大池化保留特征图中最显著的响应（最强的激活值），

        #   对于特征检测任务（边缘、纹理）更有效。

        #   平均池化会"模糊"特征，在分类任务中效果通常稍差，

        #   但在全局池化（Global Average Pooling）场景中常用。

        #

        # 为什么三次池化可以复用同一个层？

        #   因为池化层没有可学习的参数（没有 weights 和 bias），

        #   它只是一个固定的"取最大值"操作，不需要独立的参数存储。

        #   所以复用同一个 pool 对象对结果没有任何影响。

        #   相比之下，卷积层有参数，必须定义为独立的层对象，

        #   不能复用（否则三个卷积层会共享同一组参数，这不是我们想要的）。

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


        # ==============

        # Dropout 层

        # ==============

        # Dropout 在训练时随机将一部分神经元的输出置为 0，

        # 被置零的神经元这次前向传播和反向传播都不参与。

        #

        # nn.Dropout 参数详解：

        #   p=0.5 : 每个神经元被置零的概率，0.5 意味着平均丢弃一半。

        #           AlexNet 论文中使用的就是 p=0.5。

        #

        # 极其重要：Dropout 只在训练时生效！

        #   在评估（测试）时，所有神经元都参与计算，

        #   但每个神经元的输出会乘以 (1-p)，以保持期望输出一致。

        #   这个切换是通过 model.train() 和 model.eval() 控制的，

        #   nn.Module 会自动处理，但你必须记得在评估前调用 model.eval()！

        #

        # 同样地，这里也只定义一个 Dropout 层供两个全连接层复用，

        # 因为 Dropout 层也没有可学习的参数。

        self.dropout = nn.Dropout(p=0.5)


        # ==============

        # 全连接层 1（FC1）

        # ==============

        # 输入：展平后的向量，尺寸 = 256 × 4 × 4 = 4096

        # 输出：4096 维向量

        #

        # nn.Linear 参数详解：

        #   in_features=2564*4 : 输入特征维度

        #                         256 是 conv5 的输出通道数

        #                         4×4 是经过三次 pool（32→16→8→4）后的空间尺寸

        #                         这个数字必须手动计算正确，是最容易出错的地方！

        #   out_features=4096   : 输出特征维度，AlexNet 的设计选择

        #

        # 【如何快速验证 in_features 的计算是否正确？】

        # 在 forward 中 print(x.shape) 查看展平前的尺寸，

        # 或者在模型定义后用 dummy input 跑一遍：

        #   x = torch.randn(1, 3, 32, 32)

        #   x = model.features_part(x)   # 把卷积部分提出来单独跑

        #   print(x.shape)               # 看 C × H × W

        self.fc1 = nn.Linear(

            in_features=256 * 4 * 4,   # = 4096

            out_features=4096

        )


        # ==============

        # 全连接层 2（FC2）

        # ==============

        # 输入：4096 维向量

        # 输出：4096 维向量

        #

        # 这一层和 FC1 结构相同，是 AlexNet 设计中的重复结构。

        # 两个大型全连接层叠加，赋予模型强大的非线性拟合能力。

        # 代价是：这两层贡献了 AlexNet 绝大部分的参数量

        #         （4096×4096 = 16,777,216 个参数，仅这一层就有1600万个）。

        self.fc2 = nn.Linear(

            in_features=4096,

            out_features=4096

        )


        # ==============

        # 全连接层 3（FC3）—— 分类头（Classifier Head）

        # ==============

        # 输入：4096 维向量

        # 输出：num_classes 维向量（CIFAR-10 是10维）

        #

        # 输出的10个数值叫做 logits（原始分数），尚未经过 Softmax。

        # 为什么不在这里加 Softmax？

        #   因为 PyTorch 的 nn.CrossEntropyLoss 内部已经集成了 log-softmax，

        #   如果你在这里加了 Softmax，再传给 CrossEntropyLoss，

        #   就相当于对同一个数做了两次 softmax，结果是错误的。

        #   规则：配合 CrossEntropyLoss 使用时，最后一层输出原始 logits，不加任何激活函数。

        self.fc3 = nn.Linear(

            in_features=4096,

            out_features=num_classes   # CIFAR-10: 10

        )


    # ==============

    # forward 方法：定义数据的前向传播路径

    # ==============

    def forward(self, x):

        """

        前向传播方法：定义数据从输入到输出的完整流动路径。


        参数：

            x (Tensor): 输入张量，形状为 (batch_size, 3, 32, 32)

                        batch_size：一次输入的图像数量

                        3：RGB三通道

                        32×32：CIFAR-10 的图像尺寸


        返回：

            x (Tensor): 输出张量，形状为 (batch_size, 10)

                        10个类别的原始分数（logits），未经过 Softmax


        【forward 方法的调用方式】

        你永远不应该直接调用 model.forward(x)，

        而是应该像调用函数一样调用模型本身：output = model(x)。

        model(x) 会经过 nn.Module 的 call 方法，

        在调用 forward 的前后，还会自动执行 hooks（钩子函数）等操作。

        直接调用 forward 会跳过这些机制，可能导致不可预期的行为。

        """


        # ----------------------------------------------------------------

        # 卷积块 1：Conv1 → ReLU → MaxPool

        # ----------------------------------------------------------------

        # 输入 x 的形状：(batch_size, 3, 32, 32)


        # self.conv1(x) 执行卷积操作

        # 输出形状：(batch_size, 64, 32, 32)

        x = self.conv1(x)


        # F.relu：对卷积输出逐元素施加 ReLU 激活函数

        # ReLU(z) = max(0, z)，将所有负值置为0，保留正值不变。

        #

        # 【为什么卷积后必须接激活函数？】

        # 卷积是线性操作（矩阵乘法+加法），多个线性操作叠加还是线性操作。

        # 如果没有激活函数，不管叠多少层卷积，都等价于一层线性变换，

        # 无法学习非线性特征。激活函数引入非线性，是深层网络表达能力的关键。

        #

        # 【F.relu vs nn.ReLU 的区别】

        # F.relu 是函数式 API，在 forward 里直接调用，没有可学习参数，不需要在 init 定义。

        # nn.ReLU 是模块式 API，需要在 init 里实例化（self.relu = nn.ReLU()），

        # 然后在 forward 里调用（x = self.relu(x)）。

        # 两者计算结果完全一样。

        # 对于没有参数的操作（ReLU、Dropout 的函数版等），

        # 用 F.relu 更简洁，不需要在 init 里占一行。

        # 对于有参数的层（Conv、Linear），必须用 nn.Xxx 在 init 里定义。

        #

        # inplace=True 的含义：

        # 直接在 x 的内存上修改，而不是开辟新内存存放结果。

        # 好处：节省内存（对大模型很重要）。

        # 风险：如果之后需要用到修改前的 x 的值（比如残差连接），

        #       inplace 会导致原始值丢失，引发难以排查的 bug。

        #       在 AlexNet 这种简单顺序结构中没有问题，可以放心使用。

        x = F.relu(x, inplace=True)

        # 输出形状：(batch_size, 64, 32, 32)，形状不变，值域变了


        # 最大池化，空间尺寸减半：32 → 16

        # 输出形状：(batch_size, 64, 16, 16)

        x = self.pool(x)


        # ----------------------------------------------------------------

        # 卷积块 2：Conv2 → ReLU → MaxPool

        # ----------------------------------------------------------------

        # 输入形状：(batch_size, 64, 16, 16)


        x = self.conv2(x)

        # 输出形状：(batch_size, 192, 16, 16)


        x = F.relu(x, inplace=True)


        # 最大池化，空间尺寸减半：16 → 8

        # 输出形状：(batch_size, 192, 8, 8)

        x = self.pool(x)


        # ----------------------------------------------------------------

        # 卷积块 3：Conv3 → ReLU → Conv4 → ReLU → Conv5 → ReLU → MaxPool

        # ----------------------------------------------------------------

        # 这三个卷积层连续堆叠，中间不做池化，只在最后做一次池化。

        # 连续堆叠的目的：

        #   用多个小卷积核（3×3）模拟大感受野，同时参数量更少。

        #   例如，两个 3×3 卷积的感受野等同于一个 5×5 卷积，

        #   但参数量是 2×(3×3) = 18，而 5×5 = 25，参数更少。

        #   三个 3×3 的感受野等同于一个 7×7，参数比是 27 vs 49。

        # 输入形状：(batch_size, 192, 8, 8)


        x = self.conv3(x)

        # 输出形状：(batch_size, 384, 8, 8)


        x = F.relu(x, inplace=True)


        x = self.conv4(x)

        # 输出形状：(batch_size, 256, 8, 8)


        x = F.relu(x, inplace=True)


        x = self.conv5(x)

        # 输出形状：(batch_size, 256, 8, 8)


        x = F.relu(x, inplace=True)


        # 最大池化，空间尺寸减半：8 → 4

        # 输出形状：(batch_size, 256, 4, 4)

        x = self.pool(x)


        # ----------------------------------------------------------------

        # 展平（Flatten）操作：将 3D 特征图 → 1D 向量

        # ----------------------------------------------------------------

        # 到此为止，x 的形状是 (batch_size, 256, 4, 4)。

        # 全连接层（nn.Linear）要求输入是 2D 张量：(batch_size, features)，

        # 所以需要将空间维度（256, 4, 4）展平成一个向量（25644 = 4096）。

        #

        # x.view(x.size(0), -1) 的解释：

        #   x.size(0)    : 获取 batch_size 这个维度的大小，保持不变

        #   -1           : 告诉 PyTorch 自动推断这个维度的大小

        #                  PyTorch 会根据总元素数和其他维度自动计算：

        #                  总元素 = batch_size × 256 × 4 × 4

        #                  第0维固定为 batch_size

        #                  所以第1维 = 256 × 4 × 4 = 4096

        #

        # 等价写法（更现代，更语义清晰）：

        #   x = x.flatten(start_dim=1)

        #   start_dim=1 表示从第1个维度开始展平（保留第0维 batch_size 不变）

        #

        # 注意：不能写 x.view(-1, 25644)，因为这样 batch_size 维度也会被推断，

        # 虽然结果通常相同，但语义上不明确，遇到某些边缘情况可能出错。

        x = x.view(x.size(0), -1)

        # 输出形状：(batch_size, 4096)


        # ----------------------------------------------------------------

        # 全连接块 1：Dropout → FC1 → ReLU

        # ----------------------------------------------------------------

        # 注意 Dropout 放在 Linear 之前。

        # 在 AlexNet 原文中就是这样的顺序：先 Dropout，再做全连接。

        # 这样 Dropout 丢弃的是上一层的输出，

        # 等效于让全连接层的部分输入为0。

        #

        # 也有实现把 Dropout 放在 ReLU 之后（即在 FC 的输出上做 Dropout），

        # 两种放法在实践中差异不大，保持和原论文一致即可。

        x = self.dropout(x)

        # Dropout 不改变形状，只是随机将一半元素置为0

        # 形状：(batch_size, 4096)


        x = self.fc1(x)

        # 线性变换：4096 → 4096

        # 形状：(batch_size, 4096)


        x = F.relu(x, inplace=True)

        # 激活函数，形状不变：(batch_size, 4096)


        # ----------------------------------------------------------------

        # 全连接块 2：Dropout → FC2 → ReLU

        # ----------------------------------------------------------------

        # 结构与全连接块1完全相同

        x = self.dropout(x)


        x = self.fc2(x)

        # 线性变换：4096 → 4096

        # 形状：(batch_size, 4096)


        x = F.relu(x, inplace=True)

        # 形状：(batch_size, 4096)


        # ----------------------------------------------------------------

        # 输出层：FC3

        # ----------------------------------------------------------------

        # 最后一个全连接层，将 4096 维特征映射到 num_classes 维的 logits。

        # 这里不加 ReLU，也不加 Softmax。

        # 输出的 logits 将被送入 nn.CrossEntropyLoss，

        # CrossEntropyLoss 内部会先做 log_softmax 再计算交叉熵损失。

        x = self.fc3(x)

        # 输出形状：(batch_size, 10)


        return x



# ==============

# 模型结构验证（强烈建议每次定义完新模型都运行这部分）

# ==============

if name == "main":


    # 实例化模型

    model = AlexNet_CIFAR10(num_classes=10)


    # 打印模型结构：显示每一层的名称、类型和参数形状

    print("=" * 60)

    print("模型结构：")

    print(model)

    print("=" * 60)


    # 统计总参数量

    # p.numel() 返回参数张量中元素的总数

    # p.requires_grad 为 True 意味着该参数会参与梯度计算（即可学习参数）

    total_params = sum(p.numel() for p in model.parameters())

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"总参数量:     {total_params:,}")

    print(f"可训练参数量: {trainable_params:,}")

    print("=" * 60)


    # 用随机输入做一次前向传播验证形状

    # batch_size=4，模拟一次输入4张 32×32 的RGB图像

    dummy_input = torch.randn(4, 3, 32, 32)


    # model.eval() 切换到评估模式，关闭 Dropout

    # 做形状验证时用 eval 模式，避免 Dropout 随机丢弃影响我们判断

    model.eval()


    # torch.no_grad() 关闭梯度计算，节省内存，验证时不需要梯度

    with torch.no_grad():

        output = model(dummy_input)


    print(f"输入形状：{dummy_input.shape}")   # torch.Size([4, 3, 32, 32])

    print(f"输出形状：{output.shape}")         # torch.Size([4, 10])

    print("形状验证通过！")

    print("=" * 60)


    # 逐层验证中间特征图尺寸（调试利器）

    # 手动跑一遍 forward 里的每一步，打印形状变化

    print("逐层尺寸追踪：")

    model.eval()

    x = dummy_input

    with torch.no_grad():

        print(f"  输入:         {x.shape}")


        x = model.conv1(x)

        print(f"  conv1 后:     {x.shape}")    # (4, 64, 32, 32)


        x = F.relu(x)

        x = model.pool(x)

        print(f"  pool1 后:     {x.shape}")    # (4, 64, 16, 16)


        x = model.conv2(x)

        print(f"  conv2 后:     {x.shape}")    # (4, 192, 16, 16)


        x = F.relu(x)

        x = model.pool(x)

        print(f"  pool2 后:     {x.shape}")    # (4, 192, 8, 8)


        x = model.conv3(x)

        print(f"  conv3 后:     {x.shape}")    # (4, 384, 8, 8)


        x = F.relu(x)

        x = model.conv4(x)

        print(f"  conv4 后:     {x.shape}")    # (4, 256, 8, 8)


        x = F.relu(x)

        x = model.conv5(x)

        print(f"  conv5 后:     {x.shape}")    # (4, 256, 8, 8)


        x = F.relu(x)

        x = model.pool(x)

        print(f"  pool3 后:     {x.shape}")    # (4, 256, 4, 4)


        x = x.view(x.size(0), -1)

        print(f"  展平后:       {x.shape}")    # (4, 4096)


        x = model.fc1(x)

        print(f"  fc1 后:       {x.shape}")    # (4, 4096)


        x = F.relu(x)

        x = model.fc2(x)

        print(f"  fc2 后:       {x.shape}")    # (4, 4096)


        x = F.relu(x)

        x = model.fc3(x)

        print(f"  fc3(输出) 后: {x.shape}")   # (4, 10)
```

> * 完整例子

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import os


# ====================================================================
# 第一部分：全局常量定义
# ====================================================================

# CIFAR-10 的均值和标准差，提前统计好的经验值
# 定义为全局常量的好处：训练集和测试集都引用同一份，不会因为复制粘贴出错
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

# CIFAR-10 的10个类别名称，按索引顺序排列
# 类别索引 0~9 分别对应：
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


# ====================================================================
# 第二部分：模型定义（与上一节完全相同，放在这里方便整体运行）
# ====================================================================

class AlexNet_CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet_CIFAR10, self).__init__()
        self.conv1   = nn.Conv2d(3,   64,  kernel_size=3, stride=1, padding=1)
        self.conv2   = nn.Conv2d(64,  192, kernel_size=3, stride=1, padding=1)
        self.conv3   = nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1)
        self.conv4   = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.conv5   = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool    = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1     = nn.Linear(256 * 4 * 4, 4096)
        self.fc2     = nn.Linear(4096, 4096)
        self.fc3     = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x), inplace=True))
        x = self.pool(F.relu(self.conv2(x), inplace=True))
        x = F.relu(self.conv3(x), inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        x = self.pool(F.relu(self.conv5(x), inplace=True))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(self.dropout(x)), inplace=True)
        x = F.relu(self.fc2(self.dropout(x)), inplace=True)
        x = self.fc3(x)
        return x


# ====================================================================
# 第三部分：数据加载
# ====================================================================

def get_dataloaders(batch_size=128, num_workers=2):
    """
    构建并返回 CIFAR-10 的训练集和测试集 DataLoader。

    参数：
        batch_size  (int): 每个 batch 的样本数量。
                           越大训练越快，但需要更多显存。
                           128 是 CIFAR-10 上的常用值。
        num_workers (int): 数据加载的子进程数量。
                           这些子进程在 CPU 上并行做数据预处理和增强，
                           使 GPU 不必等待数据准备好再计算。
                           经验值：设为你机器 CPU 核心数的一半。
                           Windows 用户：如果报错，改为 num_workers=0。

    返回：
        train_loader: 训练集 DataLoader
        test_loader:  测试集 DataLoader
    """

    # ----------------------------------------------------------------
    # 训练集 transform：包含随机数据增强
    # ----------------------------------------------------------------
    transform_train = transforms.Compose([

        # 随机裁剪：先四周填充4像素（图像变40×40），再随机裁出32×32
        # padding_mode='reflect' 使用镜像填充，边缘更自然，
        # 比默认的 'constant'（填0黑边）质量更高
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),

        # 随机水平翻转，概率0.5
        transforms.RandomHorizontalFlip(p=0.5),

        # 转为 Tensor，同时将值域从 [0,255] 缩放到 [0.0,1.0]，
        # 维度从 HWC 转为 CHW
        transforms.ToTensor(),

        # 标准化：使每个通道的均值接近0、标准差接近1
        # 有助于梯度稳定，加速收敛
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # ----------------------------------------------------------------
    # 测试集 transform：只做确定性预处理，不做随机增强
    # ----------------------------------------------------------------
    # 测试集的目的是客观评估模型性能，必须每次产生相同的结果，
    # 所以不能有任何随机操作
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        # 注意：mean 和 std 必须与训练集一致！
    ])

    # ----------------------------------------------------------------
    # 下载并加载 CIFAR-10 数据集
    # ----------------------------------------------------------------
    # root='./data'     : 数据集存放的本地路径，第一次运行会自动下载
    # train=True/False  : True 加载训练集（50000张），False 加载测试集（10000张）
    # download=True     : 如果本地没有数据，自动从网络下载
    # transform         : 加载时自动应用的预处理 pipeline
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform_test
    )

    # ----------------------------------------------------------------
    # 构建 DataLoader
    # ----------------------------------------------------------------
    # DataLoader 的职责是：
    #   1. 按 batch_size 将数据集切分成一个个 batch
    #   2. 可选地打乱数据顺序（shuffle）
    #   3. 用多进程并行预处理数据（num_workers）
    #   4. 将数据移到固定内存（pin_memory），加速 CPU→GPU 的传输
    #
    # shuffle=True：训练时每个 epoch 打乱数据顺序。
    #   这很重要！如果不打乱，模型每次都按相同顺序见到数据，
    #   可能学到顺序相关的虚假规律，而不是真正的特征。
    #
    # shuffle=False：测试时不需要打乱，保持固定顺序便于结果对比。
    #
    # pin_memory=True：将数据加载到"固定内存"（页锁定内存），
    #   GPU 从固定内存读取数据比从普通内存快。
    #   只有在使用 GPU 时才有意义，CPU 训练时设 False 或忽略。
    #
    # drop_last=True（训练集）：如果最后一个 batch 的样本数小于 batch_size，
    #   就丢弃它。这是因为某些归一化操作（如 BatchNorm）在 batch 太小时
    #   统计量不准确，可能影响训练稳定性。测试时不需要，设 False。
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    print(f"训练集样本数：{len(train_dataset)}")   # 50000
    print(f"测试集样本数：{len(test_dataset)}")    # 10000
    print(f"训练集 batch 数：{len(train_loader)}")
    print(f"测试集 batch 数：{len(test_loader)}")

    return train_loader, test_loader


# ====================================================================
# 第四部分：单个 epoch 的训练函数
# ====================================================================

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """
    执行一个完整 epoch 的训练，即遍历一遍整个训练集。

    参数：
        model     : 要训练的模型（AlexNet_CIFAR10 实例）
        loader    : 训练集的 DataLoader
        criterion : 损失函数（nn.CrossEntropyLoss 实例）
        optimizer : 优化器（SGD 实例）
        device    : 计算设备（'cuda' 或 'cpu'）
        epoch     : 当前 epoch 编号（仅用于打印日志）

    返回：
        avg_loss  (float): 本 epoch 的平均训练 loss
        accuracy  (float): 本 epoch 的训练集准确率（0~1 之间）
    """

    # ----------------------------------------------------------------
    # 切换到训练模式
    # ----------------------------------------------------------------
    # model.train() 做了两件事：
    #   1. 启用 Dropout：训练时随机丢弃神经元
    #   2. 启用 BatchNorm 的训练行为（本模型没有BN，但养成好习惯）
    # 如果之前调用过 model.eval()，必须在训练前重新调用 model.train()，
    # 否则 Dropout 不会生效，模型不会正确训练。
    model.train()

    # 用于累加本 epoch 的总 loss 和正确预测数
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    # 记录开始时间，用于计算本 epoch 耗时
    epoch_start_time = time.time()

    # ----------------------------------------------------------------
    # 遍历训练集的每一个 batch
    # ----------------------------------------------------------------
    # enumerate(loader) 返回 (batch_index, (images, labels)) 的迭代器
    # images 形状：(batch_size, 3, 32, 32)
    # labels 形状：(batch_size,)，每个值是 0~9 的整数类别索引
    for batch_idx, (images, labels) in enumerate(loader):

        # 将数据移到目标设备（CPU 或 GPU）
        # .to(device) 是非原地操作，必须用赋值接收返回值
        # 如果 device 是 'cuda'，这一步将数据从内存复制到显存
        images = images.to(device)
        labels = labels.to(device)

        # ------------------------------------------------------------
        # 核心五步：PyTorch 训练的标准流程，每个 batch 都要执行
        # ------------------------------------------------------------

        # 【第一步】清空梯度
        # 必 原因：PyTorch 默认会将每次 backward 的梯度累加到 .grad 属性中。
        # 如果不清空，第二个 batch 的梯度会叠加到第一个 batch 的梯度上，
        # 导致参数更新方向错误，模型无法正常收敛。
        # 唯一例外：梯度累加（Gradient Accumulation）是故意不清零，
        # 用于模拟更大的 batch size，但那是高级技巧，初学阶段先不管。
        optimizer.zero_grad()

        # 【第二步】前向传播
        # 将一个 batch 的图像送入模型，得到预测的 logits
        # outputs 形状：(batch_size, 10)
        # outputs[i][j] 表示第 i 张图属于第 j 类的原始分数（未经 softmax）
        outputs = model(images)

        # 【第三步】计算损失
        # criterion 是 nn.CrossEntropyLoss 实例
        # 它接收两个参数：
        #   - outputs：模型输出的 logits，形状 (batch_size, num_classes)
        #   - labels：真实类别索引，形状 (batch_size,)，值为 0~9 的整数
        # 内部计算：先对 outputs 做 log_softmax，再计算负对数似然损失
        # 返回：这个 batch 中所有样本 loss 的平均值（标量 Tensor）
        #
        # 【注意】CrossEntropyLoss 要求 labels 是类别索引（整数），
        # 不是 one-hot 编码！如果你的标签是 one-hot 形式，需要先转换。
        loss = criterion(outputs, labels)

        # 【第四步】反向传播
        # loss.backward() 从 loss 出发，利用链式法则，
        # 自动计算 loss 对模型所有可学习参数的梯度，
        # 并将梯度累加到各参数的 .grad 属性中。
        # 这一步是深度学习的核心，PyTorch 的自动微分（autograd）在这里发挥作用。
        loss.backward()

        # 【第五步】更新参数
        # optimizer.step() 根据各参数的 .grad 和优化算法（SGD/Adam等），
        # 计算参数更新量并修改参数值。
        # SGD 的更新公式：
        #   θ ← θ - lr × ∇θL
        # 带动量的 SGD：
        #   v ← momentum × v + ∇θL
        #   θ ← θ - lr × v
        optimizer.step()

        # ------------------------------------------------------------
        # 统计本 batch 的 loss 和准确率
        # ------------------------------------------------------------

        # loss.item()：将单元素 Tensor 转为 Python float。
        # 必须用 .item() 而不是直接用 loss！
        # 原因：loss 是一个带有计算图的 Tensor，如果直接累加 Tensor，
        # 会保留整个计算图的引用，导致内存持续增长，最终 OOM。
        # .item() 取出纯数值，切断与计算图的联系。
        total_loss += loss.item()

        # outputs.argmax(dim=1)：
        #   对每个样本，在10个类别分数中取最大值的索引，作为预测类别。
        #   dim=1 表示在第1个维度（类别维度）上取 argmax。
        #   结果形状：(batch_size,)
        #
        # == labels：逐元素比较预测类别和真实类别，相同为 True，否则 False
        #   结果形状：(batch_size,)，bool 类型
        #
        # .sum().item()：统计 True 的数量（即预测正确的样本数），转为 Python int
        predicted = outputs.argmax(dim=1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)  # labels.size(0) 即当前 batch 的样本数

        # 每 100 个 batch 打印一次进度
        if (batch_idx + 1) % 100 == 0:
            # 当前已处理样本数
            processed = (batch_idx + 1) * loader.batch_size
            print(f"  Epoch {epoch} | Batch [{batch_idx+1}/{len(loader)}] "
                  f"| 已处理 {processed}/{len(loader.dataset)} 样本 "
                  f"| 当前 batch loss: {loss.item():.4f}")

    # 计算整个 epoch 的平均 loss 和准确率
    avg_loss = total_loss / len(loader)        # 总loss除以batch数
    accuracy = total_correct / total_samples   # 正确数除以总样本数

    epoch_time = time.time() - epoch_start_time

    print(f"  → Epoch {epoch} 训练完成 | "
          f"耗时: {epoch_time:.1f}s | "
          f"avg loss: {avg_loss:.4f} | "
          f"train acc: {accuracy*100:.2f}%")

    return avg_loss, accuracy


# ====================================================================
# 第五部分：验证/测试函数
# ====================================================================

def evaluate(model, loader, criterion, device):
    """
    在给定数据集（通常是测试集）上评估模型性能。

    参数：
        model     : 要评估的模型
        loader    : 测试集的 DataLoader
        criterion : 损失函数
        device    : 计算设备

    返回：
        avg_loss  (float): 测试集平均 loss
        accuracy  (float): 测试集准确率（0~1 之间）
    """

    # ----------------------------------------------------------------
    # 切换到评估模式（极其重要，最常见的易错点之一）
    # ----------------------------------------------------------------
    # model.eval() 做了两件事：
    #   1. 关闭 Dropout：所有神经元都参与计算，但输出乘以 (1-p) 保持期望一致
    #   2. BatchNorm 切换为使用训练时统计的全局均值/方差，而非当前batch的统计量
    #
    # 如果忘记调用 model.eval()，Dropout 会在测试时随机丢弃神经元，
    # 导致：
    #   - 每次评估结果不同（随机性）
    #   - 准确率偏低（有效神经元减少了）
    model.eval()

    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    # ----------------------------------------------------------------
    # torch.no_grad()：禁用梯度计算
    # ----------------------------------------------------------------
    # 在评估阶段，我们只做前向传播，不需要计算梯度。
    # torch.no_grad() 的作用：
    #   1. 不构建计算图（不追踪操作历史），节省大量内存
    #   2. 跳过梯度相关计算，加快前向传播速度（通常快约30%）
    # 用法：作为上下文管理器（with 语句），其作用域内所有操作都不计算梯度。
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播（没有 optimizer.zero_grad 和 loss.backward，因为不更新参数）
            outputs = model(images)

            # 计算 loss（用于监控，判断模型是否过拟合）
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # 统计正确预测数
            predicted = outputs.argmax(dim=1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


# ====================================================================
# 第六部分：模型保存与加载工具函数
# ====================================================================

def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, filepath):
    """
    保存训练检查点（Checkpoint）。

    检查点包含恢复训练所需的全部状态，使训练可以从中断处继续，
    而不必从头开始。

    参数：
        model     : 模型实例
        optimizer : 优化器实例
        scheduler : 学习率调度器实例
        epoch     : 当前 epoch 编号
        best_acc  : 目前为止最好的测试准确率
        filepath  : 保存路径（.pth 文件）
    """

    # ----------------------------------------------------------------
    # 什么是 state_dict？
    # ----------------------------------------------------------------
    # state_dict 是一个 Python 字典，将层的名称映射到对应的参数张量。
    # 对于模型，包含所有可学习参数（weights 和 bias）。
    # 对于优化器，包含动量缓冲、学习率等优化状态。
    #
    # 为什么不直接保存整个模型对象（torch.save(model, ...)）？
    # 保存整个模型对象依赖于代码文件的路径和类定义，
    # 如果你移动了代码文件或重命名了类，加载会失败。
    # 只保存 state_dict 更鲁棒、更通用，是官方推荐做法。
    checkpoint = {
        'epoch'          : epoch,
        'model_state'    : model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_acc'       : best_acc,
    }
    torch.save(checkpoint, filepath)
    print(f"  ✓ 检查点已保存到 {filepath}")


def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device='cpu'):
    """
    加载训练检查点，恢复模型和优化器状态。

    参数：
        filepath  : 检查点文件路径
        model     : 模型实例（必须和保存时的结构相同）
        optimizer : 优化器实例（如果只做推理可以不传）
        scheduler : 学习率调度器实例（可选）
        device    : 加载到哪个设备

    返回：
        start_epoch (int)  : 下一个要训练的 epoch 编号
        best_acc    (float): 历史最佳准确率
    """

    # map_location=device：将检查点中保存的张量加载到指定设备。
    # 比如模型在 GPU 上训练保存的，在 CPU 上加载时要指定 map_location='cpu'，
    # 否则 PyTorch 会尝试加载到原来的 GPU，如果该 GPU 不存在则报错。
    checkpoint = torch.load(filepath, map_location=device)

    # 将保存的参数恢复到模型中
    # strict=True（默认）：要求 checkpoint 的 key 和模型的 key 完全一致
    # strict=False：允许部分加载（迁移学习时常用，加载预训练权重的子集）
    model.load_state_dict(checkpoint['model_state'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state'])

    start_epoch = checkpoint['epoch'] + 1   # 从下一个 epoch 继续
    best_acc    = checkpoint['best_acc']

    print(f"  ✓ 检查点加载成功，从 epoch {start_epoch} 继续训练，历史最佳 acc: {best_acc*100:.2f}%")

    return start_epoch, best_acc


# ====================================================================
# 第七部分：主训练函数
# ====================================================================

def train(
    num_epochs   = 90,
    batch_size   = 128,
    learning_rate= 0.01,
    weight_decay = 5e-4,
    momentum     = 0.9,
    num_workers  = 2,
    save_dir     = './checkpoints',
    resume       = None       # 如果不为None，从这个路径的检查点恢复训练
):
    """
    完整的训练主函数，包含训练循环、验证、学习率调度、检查点保存。

    参数：
        num_epochs    : 总训练轮数
        batch_size    : 每个 batch 的样本数
        learning_rate : 初始学习率
        weight_decay  : L2 正则化系数
        momentum      : SGD 动量系数
        num_workers   : DataLoader 的工作进程数
        save_dir      : 检查点保存目录
        resume        : 检查点路径，不为 None 时从断点恢复训练
    """

    # ----------------------------------------------------------------
    # 设备检测
    # ----------------------------------------------------------------
    # torch.cuda.is_available()：检测当前机器是否有可用的 NVIDIA GPU
    # 如果有 GPU，使用 'cuda'；否则使用 'cpu'
    # MPS 是 Apple Silicon (M1/M2) 的 GPU 后端，也检测一下
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"使用设备：{device}")
    if device.type == 'cuda':
        # 打印 GPU 型号，方便确认
        print(f"GPU 型号：{torch.cuda.get_device_name(0)}")

    # ----------------------------------------------------------------
    # 准备数据
    # ----------------------------------------------------------------
    train_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers
    )

    # ----------------------------------------------------------------
    # 初始化模型并移到目标设备
    # ----------------------------------------------------------------
    model = AlexNet_CIFAR10(num_classes=10)

    # .to(device)：将模型的所有参数和缓冲区移到目标设备。
    # 这是非原地操作，必须用赋值接收（model = model.to(device)），
    # 或者直接写 model.to(device)（因为 to() 同时也会原地修改模型）。
    # 两种写法都可以，后者更简洁。
    model = model.to(device)

    # ----------------------------------------------------------------
    # 定义损失函数
    # ----------------------------------------------------------------
    # nn.CrossEntropyLoss 是分类任务的标准损失函数。
    # 它在内部做了以下三步：
    #   1. 对 logits 做 softmax：将原始分数转为概率分布
    #   2. 取对数：log(softmax(logits))
    #   3. 取真实类别对应的对数概率，取负号：-log(p_true)
    # 这个值越小，说明模型对正确类别的预测概率越高。
    #
    # reduction='mean'（默认）：对 batch 内所有样本的 loss 取平均。
    # 也可以用 'sum'（求和）或 'none'（返回每个样本的 loss）。
    criterion = nn.CrossEntropyLoss(reduction='mean')

    # ----------------------------------------------------------------
    # 定义优化器
    # ----------------------------------------------------------------
    # optim.SGD：随机梯度下降优化器，是深度学习中最经典的优化器。
    #
    # 参数详解：
    #   model.parameters() : 告诉优化器要更新哪些参数（所有可学习参数）
    #   lr=learning_rate   : 学习率，控制每次参数更新的步长
    #                        过大：loss 震荡不收敛；过小：收敛极慢
    #   momentum=0.9       : 动量系数，加速收敛并减少震荡
    #                        物理直觉：参数更新有"惯性"，
    #                        在梯度方向一致时会加速，方向变化时会减缓
    #                        0.9 是非常常用的经验值
    #   weight_decay=5e-4  : L2 正则化系数，在参数更新时对大参数施加惩罚
    #                        等价于在 loss 上加 λ/2 * ||θ||²
    #                        防止参数值过大导致过拟合
    #                        5e-4 是 AlexNet 原文的设置
    #
    # 为什么不用 Adam？
    #   Adam 自适应学习率，通常收敛更快，但在 CIFAR-10 上 SGD+动量
    #   经过足够 epoch 训练后，最终精度通常比 Adam 略高。
    #   这是因为 SGD 的正则化效果更好，泛化能力更强。
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay
    )

    # ----------------------------------------------------------------
    # 定义学习率调度器
    # ----------------------------------------------------------------
    # 学习率调度器在训练过程中自动调整学习率。
    # 为什么要调整学习率？
    #   训练初期：大学习率快速找到大致的参数范围
    #   训练后期：小学习率精细调整，找到更好的局部最优
    #
    # StepLR 参数详解：
    #   optimizer  : 要调整学习率的优化器
    #   step_size=30: 每隔30个epoch，将学习率乘以 gamma
    #   gamma=0.1  : 学习率衰减因子
    #                第 0  个 epoch：lr = 0.01
    #                第 30 个 epoch：lr = 0.01 × 0.1 = 0.001
    #                第 60 个 epoch：lr = 0.001 × 0.1 = 0.0001
    #                第 90 个 epoch：lr = 0.0001 × 0.1 = 0.00001
    #
    # 注意：scheduler.step() 必须在每个 epoch 结束后调用，
    # 不是每个 batch 后！（这是一个常见错误）
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=30,
        gamma=0.1
    )

    # ----------------------------------------------------------------
    # 从检查点恢复（可选）
    # ----------------------------------------------------------------
    start_epoch = 1
    best_acc    = 0.0

    # 创建检查点保存目录
    os.makedirs(save_dir, exist_ok=True)

    if resume is not None and os.path.isfile(resume):
        start_epoch, best_acc = load_checkpoint(
            resume, model, optimizer, scheduler, device
        )

    # ----------------------------------------------------------------
    # 用于记录训练历史，方便后续画图分析
    # ----------------------------------------------------------------
    history = {
        'train_loss': [],
        'train_acc' : [],
        'test_loss' : [],
        'test_acc'  : [],
        'lr'        : [],
    }

    # ----------------------------------------------------------------
    # 训练主循环
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"开始训练，共 {num_epochs} 个 epoch")
    print("=" * 60)

    total_start_time = time.time()

    for epoch in range(start_epoch, num_epochs + 1):

        # 打印当前学习率
        # optimizer.param_groups 是一个列表，每个元素是一组参数的配置
        # 通常只有一组，所以取 [0]['lr']
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch [{epoch}/{num_epochs}]  当前学习率: {current_lr:.6f}")

        # ---- 训练一个 epoch ----
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # ---- 在测试集上评估 ----
        # 注意：evaluate 内部会调用 model.eval()，
        # 训练前记得 train_one_epoch 里已经调用了 model.train()，
        # 两者交替调用，互不干扰
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        # ---- 更新学习率调度器 ----
        # 必须在 evaluate 之后、下一个 epoch 开始之前调用
        # scheduler.step() 内部会将 epoch 计数器加1，
        # 并根据 step_size 判断是否要更新学习率
        scheduler.step()

        # ---- 记录历史 ----
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['lr'].append(current_lr)

        # ---- 打印本 epoch 的汇总结果 ----
        print(f"  ▶ 汇总 | "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc*100:.2f}%  | "
              f"Test Loss:  {test_loss:.4f}  Test Acc:  {test_acc*100:.2f}%")

        # ---- 诊断过拟合 ----
        # 训练准确率和测试准确率差距过大，说明过拟合
        overfit_gap = train_acc - test_acc
        if overfit_gap > 0.15:
            print(f"  ⚠ 注意：训练/测试准确率差距 {overfit_gap*100:.1f}%，可能存在过拟合")

        # ---- 保存最优模型 ----
        # 每次测试准确率创新高时，保存当前模型为 best_model.pth
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_acc,
                filepath=os.path.join(save_dir, 'best_model.pth')
            )
            print(f"  ★ 新的最佳测试准确率：{best_acc*100:.2f}%")

        # ---- 每10个epoch保存一个常规检查点 ----
        # 即使这不是最优模型，也定期保存，防止训练意外中断
        if epoch % 10 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_acc,
                filepath=os.path.join(save_dir, f'checkpoint_epoch{epoch}.pth')
            )

    # ---- 训练结束 ----
    total_time = time.time() - total_start_time
    print("\n" + "=" * 60)
    print(f"训练完成！")
    print(f"总耗时：{total_time/60:.1f} 分钟")
    print(f"最佳测试准确率：{best_acc*100:.2f}%")
    print("=" * 60)

    return model, history


# ====================================================================
# 第八部分：训练结果可视化
# ====================================================================

def plot_history(history, save_path='./training_history.png'):
    """
    将训练历史（loss 曲线和准确率曲线）绘制成图像并保存。

    参数：
        history   : train() 函数返回的 history 字典
        save_path : 图像保存路径
    """
    import matplotlib.pyplot as plt

    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ---- 子图1：Loss 曲线 ----
    axes[0].plot(epochs, history['train_loss'], label='Train Loss', color='blue')
    axes[0].plot(epochs, history['test_loss'],  label='Test Loss',  color='red')
    axes[0].set_title('Loss Curve')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ---- 子图2：准确率曲线 ----
    # 将 0~1 的准确率转为百分比显示，更直观
    train_acc_pct = [a * 100 for a in history['train_acc']]
    test_acc_pct  = [a * 100 for a in history['test_acc']]

    axes[1].plot(epochs, train_acc_pct, label='Train Acc', color='blue')
    axes[1].plot(epochs, test_acc_pct,  label='Test Acc',  color='red')
    axes[1].set_title('Accuracy Curve')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # ---- 子图3：学习率变化 ----
    axes[2].plot(epochs, history['lr'], color='green')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_yscale('log')   # 对数坐标更清晰地展示学习率的阶梯式下降
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"训练曲线已保存到 {save_path}")


# ====================================================================
# 第九部分：单张图像预测（推理演示）
# ====================================================================

def predict_single_image(model, image_tensor, device):
    """
    对单张图像做预测，返回预测类别和各类别的概率。

    参数：
        model        : 已训练好的模型
        image_tensor : 经过预处理的图像 Tensor，形状 (3, 32, 32)
                       注意：必须已经做过 ToTensor 和 Normalize
        device       : 计算设备

    返回：
        pred_class (str)        : 预测的类别名称
        probabilities (Tensor)  : 10个类别的概率，形状 (10,)
    """

    model.eval()

    # unsqueeze(0)：在第0维增加一个维度，将 (3,32,32) 变为 (1,3,32,32)
    # 因为模型期望输入是 (batch_size, C, H, W)，即使只有一张图也要有 batch 维度
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)       # 形状：(1, 10)

        # F.softmax：将 logits 转为概率分布（各类别概率之和为1）
        # dim=1：在类别维度上做 softmax
        # squeeze(0)：去掉 batch 维度，变为 (10,)
        probabilities = F.softmax(logits, dim=1).squeeze(0)

        # 取概率最大的类别索引
        pred_idx = probabilities.argmax().item()

    pred_class = CIFAR10_CLASSES[pred_idx]
    confidence = probabilities[pred_idx].item()

    print(f"预测类别：{pred_class}（置信度：{confidence*100:.1f}%）")
    print("各类别概率：")
    for i, (cls, prob) in enumerate(zip(CIFAR10_CLASSES, probabilities)):
        bar = '█' * int(prob.item() * 30)   # 用 ASCII 字符画简单的概率条
        print(f"  {cls:12s} {prob.item()*100:5.1f}%  {bar}")

    return pred_class, probabilities


# ====================================================================
# 程序入口
# ====================================================================

if __name__ == "__main__":

    # ----------------------------------------------------------------
    # 启动训练
    # ----------------------------------------------------------------
    # 如果想从检查点恢复训练，修改 resume 参数：
    # resume='./checkpoints/checkpoint_epoch30.pth'
    model, history = train(
        num_epochs    = 90,
        batch_size    = 128,
        learning_rate = 0.01,
        weight_decay  = 5e-4,
        momentum      = 0.9,
        num_workers   = 2,          # Windows 用户改为 0
        save_dir      = './checkpoints',
        resume        = None        # 首次训练填 None
    )

    # ----------------------------------------------------------------
    # 绘制训练曲线
    # ----------------------------------------------------------------
    plot_history(history, save_path='./training_history.png')

    # ----------------------------------------------------------------
    # 演示：对测试集中的一张图像做推理
    # ----------------------------------------------------------------
    # 加载测试集（只用于取一张图做演示）
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=False, transform=transform_test
    )

    # 取第0张图像和它的真实标签
    sample_image, true_label = test_dataset[0]

    print(f"\n真实类别：{CIFAR10_CLASSES[true_label]}")

    device = next(model.parameters()).device  # 获取模型当前所在的设备
    predict_single_image(model, sample_image, device)

```

```arduino
准备工作（只做一次）
  ↓
加载数据集，定义模型、损失函数、优化器、学习率调度器
  ↓
┌─────────────────────────────────────────────┐
│  epoch 循环（重复 90 次）                    │
│                                             │
│  ① 训练一个 epoch                           │
│     └─ 遍历训练集的每一个 batch              │
│         对每个 batch 执行：                  │
│         清空梯度 → 前向传播 → 计算loss       │
│         → 反向传播 → 更新参数               │
│                                             │
│  ② 在测试集上评估（不更新参数）              │
│     └─ 遍历测试集，统计 loss 和准确率        │
│                                             │
│  ③ 学习率调度器 step 一次                   │
│     └─ 判断是否到了该降低学习率的 epoch      │
│                                             │
│  ④ 判断是否是历史最佳准确率                  │
│     └─ 是 → 立刻保存为 best_model.pth      │
│     └─ 否 → 不保存（或每10个epoch存一次）   │
└─────────────────────────────────────────────┘
  ↓
训练结束，best_model.pth 里就是最好的模型

```

看结果，有点过拟合。根本原因是 AlexNet 对 CIFAR-10 来说**模型容量偏大**。AlexNet 有大约 5800 万个参数，而 CIFAR-10 只有 50000 张训练图，参数数量远超样本数量，模型有足够的"记忆力"把训练集背下来。

此外用的数据增强比较基础，只有随机裁剪和水平翻转，对这个量级的模型来说约束力不够强

![image-20260225215138901](F:\note\deep_learning\pytorch_learning\day08_AlexNet.assets\image-20260225215138901.png)

 > * Checkpoint 是什么

Checkpoint 直译是"存档点"，你可以完全用游戏存档来理解它。

想象你在打一个需要 10 小时通关的游戏，你不会一口气打完，你会每隔一段时间手动存一次档。如果中途断电或者游戏崩了，你可以从上一个存档继续，而不用从头开始。Checkpoint 就是这个作用。

具体到代码里，训练 90 个 epoch 在普通机器上可能需要几个小时。如果训练到第 60 个 epoch 时服务器断了，没有 Checkpoint 的话前面所有的训练都白费了。

一个 Checkpoint 文件（`.pth`）里保存了以下内容：

```
checkpoint = {
    'epoch'          : 当前训练到第几个epoch，
    'model_state'    : 模型所有层的参数（weights和bias），
    'optimizer_state': 优化器的内部状态（动量缓冲等），
    'scheduler_state': 学习率调度器的内部状态（当前到第几步了），
    'best_acc'       : 目前为止最好的测试准确率，
}
```

其中最重要的是 `model_state`，也就是模型参数本身。其他三个是为了让训练能够**无缝续接**而保存的额外状态。

------

> * 为什么光保存模型参数不够，还要保存优化器和调度器的状态？

举个例子说明。假设你训练到 epoch 50 时中断了，你重新加载模型参数接着训练：

**只恢复模型参数的问题**：优化器（SGD with momentum）内部维护着每个参数的"动量缓冲"，记录着过去梯度的运动方向。这个缓冲是训练了 50 个 epoch 积累下来的。如果不恢复它，优化器就像"失忆"了一样，需要重新积累动量，前几个 epoch 的参数更新方向会不准确。

**只恢复模型参数和优化器的问题**：学习率调度器记录着"我已经走到第 50 步了"。如果不恢复它，调度器会认为自己从第 0 步开始，学习率会重置回初始值 0.01，而正确的值应该是已经衰减后的 0.001。用错误的学习率继续训练，模型很可能跑偏。

所以三个状态要一起保存、一起恢复，才能做到真正意义上的"从断点无缝续接"。

------

> * 代码里保存了几种文件，有什么区别

代码里保存了两类文件，目的不同：

**`best_model.pth`**：只要测试准确率创新高就覆盖保存。训练全程只有一个，始终是当前最好的模型。训练结束后拿这个文件做推理和部署。

**`checkpoint_epoch30.pth`、`checkpoint_epoch60.pth`……**：每隔 10 个 epoch 保存一次，主要用于防止训练意外中断后从头来过。这些文件平时用不到，但关键时刻能救命。训练顺利完成后可以删掉。

## 02.1 小结

AlexNet 作为一个完整的工程方案早已过时，但它贡献的若干**设计思想**至今仍是现代深度学习的基础。

> * ReLU 激活函数是它最重要的遗产。用 max(0, x))替代 Sigmoid 解决了深层网络的梯度消失问题，这个选择在今天几乎成了默认标准，你在任何现代网络里都能看到它的身影，或者它的变体（LeakyReLU、GELU 等）

> * Dropout 正则化的思想同样延续至今。虽然现代网络更多依赖 BatchNorm 和数据增强来防止过拟合，但"训练时随机丢弃，推理时全员参与"这个核心思路在 Transformer 等架构中仍被广泛使用。

> * 数据增强是 AlexNet 另一个至今都不过时的贡献。随机裁剪、水平翻转这套基础 pipeline 你在 ResNet、ViT 等任何现代模型的训练代码里都能原样找到，几乎没有变化。

==**卷积 + 池化 + 全连接**的三段式结构（特征提取 → 下采样 → 分类头）奠定了 CNN 的基本范式。后来的 VGG、ResNet 都是在这个骨架上演进，理解了 AlexNet 的结构逻辑，再看那些模型会顺畅很多==

最后一点是**工程意识**。AlexNet 把模型设计、正则化策略、数据处理作为一个整体来思考，而不是孤立地堆叠层数。这种"网络结构 + 训练策略需要协同设计"的意识，在今天做任何深度学习项目时都值得记住。

------

总结成一句话：**AlexNet 的具体参数和结构已经过时，但它回答的那些问题——如何防止过拟合、如何让深层网络稳定训练、如何扩充数据——在今天仍然是每个模型都要面对的核心问题。**



**疑惑**：数据增强是不是像"复印机"一样，把原始数据集提前生成多份变体，从而物理上扩大了数据集的数量？如果是这样，batch size 等设置不就都要跟着变了？

**解答**：不是。PyTorch 的数据增强是**在线增强**，发生在 DataLoader 每次取一张图的瞬间，而不是提前生成好存起来。硬盘上的数据集数量始终不变，batch size 等设置完全不受影响。

**增强发生的精确位置**：DataLoader 内部遍历数据集时，会调用 `dataset.__getitem__(index)`，在这个方法里，原始图片被读出后立即经过 `transform` 处理，`transform` 里所有 `Random` 开头的操作在此刻执行，产生一个随机变体，直接送入模型，不保存。

**增强的效果体现在哪里**：体现在多个 epoch 的累积上。同一张图在第 1 个 epoch 被取出时随机翻转，第 2 个 epoch 被取出时随机裁剪，每次结果不同。训练 90 个 epoch，模型实际见过了这张图的大量不同面貌，效果等价于"把数据集扩大了很多倍"，只是这些变体不是提前生成的，而是分散在每个 epoch 里即时产生的。

**一句话记住**：数据增强不扩大数据集，它让同一张图在每个 epoch 里"长得不一样"









