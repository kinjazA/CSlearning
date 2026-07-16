# 01 学习率衰减方法

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import numpy as np

# 1. 设置随机种子，保证每次运行结果一致
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 第一步：数据准备 (Data Preparation)
# ==========================================
print("正在生成数据...")

# 模拟 2000 个样本，每个样本有 5 个特征 (x1, x2, x3, x4, x5)
n_samples = 2000
n_features = 5
X_numpy = np.random.randn(n_samples, n_features)

# 定义一个复杂的非线性关系：y = x1^2 + sin(x2) + exp(0.5*x3) - x4*x5
# 这是一个典型的“多元非线性回归”问题
y_numpy = (X_numpy[:, 0]**2) + np.sin(X_numpy[:, 1]) + np.exp(0.5 * X_numpy[:, 2]) - (X_numpy[:, 3] * X_numpy[:, 4])
# 加一点随机噪声，模拟真实世界的误差
y_numpy += 0.1 * np.random.randn(n_samples)

# 转换为 PyTorch 的 Tensor 类型
X_tensor = torch.tensor(X_numpy, dtype=torch.float32)
y_tensor = torch.tensor(y_numpy, dtype=torch.float32).unsqueeze(1) # 变成 (2000, 1)

# 划分训练集 (80%) 和 测试集 (20%)
train_size = int(0.8 * n_samples)
train_dataset = TensorDataset(X_tensor[:train_size], y_tensor[:train_size])
test_dataset = TensorDataset(X_tensor[train_size:], y_tensor[train_size:])

# 创建数据加载器 (Batch Size = 64)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ==========================================
# 第二步：模型搭建 (Model Building) - 不使用 nn.Sequential
# ==========================================
class MyRegressionNet(nn.Module):
    def __init__(self):
        super(MyRegressionNet, self).__init__()
        
        # --- 在这里定义所有的层 ---
        # 输入层到隐藏层1: 输入5维 -> 输出64维
        self.hidden_layer1 = nn.Linear(5, 64)
        
        # 隐藏层1到隐藏层2: 输入64维 -> 输出32维
        self.hidden_layer2 = nn.Linear(64, 32)
        
        # 隐藏层2到输出层: 输入32维 -> 输出1维 (预测值)
        self.output_layer = nn.Linear(32, 1)
        
        # 激活函数层 (这里实例化为层，也可以在forward里直接用F.relu)
        self.relu = nn.ReLU()

    def forward(self, x):
        # --- 在这里按顺序连接层 ---
        
        # 第一层计算：线性变换 -> 激活
        x = self.hidden_layer1(x)
        x = self.relu(x)
        
        # 第二层计算
        x = self.hidden_layer2(x)
        x = self.relu(x)
        
        # 输出层计算 (回归问题最后一层通常不用激活函数)
        x = self.output_layer(x)
        
        return x

model = MyRegressionNet()
print("模型结构如下：")
print(model)

# ==========================================
# 第三步：配置训练 (Training Setup)
# ==========================================
# 损失函数：均方误差 (MSE)
criterion = nn.MSELoss()

# 优化器：使用随机梯度下降 (SGD)，初始学习率设为 0.1
optimizer = optim.SGD(model.parameters(), lr=0.1)

# ---------------------------------------------------------
# 【重点】学习率衰减策略选择 (修改这里的变量来切换)
# ---------------------------------------------------------
# 可选模式: 'step', 'cosine', 'plateau'
scheduler_mode = 'step' 

print(f"\n正在配置学习率调度器: {scheduler_mode}")

if scheduler_mode == 'step':
    # 策略 A: 等间隔衰减 (StepLR)
    # 每 20 个 epoch，学习率乘以 gamma 0.1
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

elif scheduler_mode == 'cosine':
    # 策略 B: 余弦退火 (CosineAnnealingLR)
    # 学习率像余弦波浪一样平滑下降，直到 epoch 结束
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)

elif scheduler_mode == 'plateau':
    # 策略 C: 遇到瓶颈自适应衰减 (ReduceLROnPlateau)
    # 如果 Loss 在 5 个 epoch 内不下降，学习率乘以 0.5
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# ==========================================
# 第四步：训练循环 (Training Loop)
# ==========================================
epochs = 60
loss_history = []  # 记录每轮的 Loss
lr_history = []    # 记录每轮的 学习率

print("\n开始训练...")

for epoch in range(epochs):
    # --- 1. 训练阶段 ---
    model.train() # 切换到训练模式
    epoch_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        # 清空梯度
        optimizer.zero_grad()
        
        # 前向传播
        predictions = model(batch_X)
        
        # 计算损失
        loss = criterion(predictions, batch_y)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        epoch_loss += loss.item()
    
    # 计算本轮平均 Loss
    avg_train_loss = epoch_loss / len(train_loader)
    loss_history.append(avg_train_loss)
    
    # 记录当前学习率
    current_lr = optimizer.param_groups[0]['lr']
    lr_history.append(current_lr)
    
    # --- 2. 更新学习率 (Scheduler Step) ---
    # 注意：Plateau 调度器需要传入 Loss 值，其他的不需要
    if scheduler_mode == 'plateau':
        scheduler.step(avg_train_loss)
    else:
        scheduler.step()
    
    # 每 5 轮打印一次进度
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_train_loss:.4f} | LR: {current_lr:.6f}")

# ==========================================
# 第五步：评估与可视化 (Evaluation & Visualization)
# ==========================================
model.eval() # 切换到评估模式
with torch.no_grad():
    # 在测试集上预测一次看看效果
    test_X, test_y = test_dataset[:]
    predicted = model(test_X)
    test_loss = criterion(predicted, test_y)
    print(f"\n最终测试集 MSE Loss: {test_loss.item():.4f}")

# 画图
plt.figure(figsize=(12, 5))

# 子图 1: Loss 变化
plt.subplot(1, 2, 1)
plt.plot(loss_history, label='Training Loss', color='red')
plt.title(f'Loss Curve ({scheduler_mode})')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.yscale('log') # 用对数坐标轴，看后期下降更清楚
plt.grid(True)
plt.legend()

# 子图 2: 学习率变化
plt.subplot(1, 2, 2)
plt.plot(lr_history, label='Learning Rate', color='blue', linestyle='--')
plt.title(f'Learning Rate Decay ({scheduler_mode})')
plt.xlabel('Epoch')
plt.ylabel('LR')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
```

![image-20260210133312637](F:\note\deep_learning\pytorch_learning\day04_学习率衰减方法.assets\image-20260210133312637.png)

# 02 正则化方法

## 02.1 随机失活Droupout

> *  1. 训练阶段 (Training)：为了防止“抱大腿”

想象一家创业公司（神经网络），有一个超级明星员工叫“老王”（某个高权重的神经元）。

- **没有 Dropout 时**：因为老王能力太强，每次遇到难题（Loss），大家都等着老王解决。其他员工（其他神经元）发现只要跟着老王混就能完成任务，于是他们开始偷懒，不再独立思考，甚至专门配合老王的怪癖（**这就是过拟合 / 共适应 Co-adaptation**）
- **有了 Dropout 时**：CEO 制定了一条变态规定——**每天早上抽签，随机让 50% 的员工强制休假，包括老王**
- **结果**：
   - 老王今天不在？其他人必须顶上
   - 因为不知道谁明天会休假，**每个人都必须具备独当一面的能力**，不能依赖任何人
   - 最终，团队里的每个人都变得很强壮（特征提取能力很强）

> *  2. 测试/推理阶段 (Testing)：为了“火力全开”

现在公司要上市了，面临最终的大考（测试集）

- **这时候你还会让 50% 的人休假吗？** 当然不会
- 你会把所有培养出来的精兵强将**全部派上场**
- 因为每个人在训练时都被逼成了多面手，现在所有人合力，效果自然是 1+1 > 2 的

> * ### 初始状态 vs. 训练状态
>
>    - **刚初始化时：**
>        在代码刚跑起来时，权重都是随机初始化的（比如全是高斯分布的随机数）。这时候，神经元 A 和神经元 B 确实没啥区别，都是“白纸”。这时候你删谁都一样
>
>    - **训练开始后：**
>        一旦开始反向传播，**分化（Differentiation）** 就开始了
>
>       - 神经元 A 可能运气好，刚好对“圆形的轮廓”有反应，梯度下降会让它越来越擅长识别“圆”
>       - 神经元 B 可能对“红色的纹理”有反应，它会越来越擅长识别“红”
>
>       **此时：**
>
>       - **随机失活 A (圆)** → 网络只能看到“红色”，它必须学会仅凭颜色猜出这是个苹果
>       - **随机失活 B (红)** →网络只能看到“圆形”，它必须学会仅凭形状猜出这是个苹果
>
>       **结论：** 失活 A 和失活 B，丢掉的是**完全不同的信息特征**。正是这种“不同的缺失”，强迫网络学会了完整的、多维度的判断能力

> * **Dropout 加在哪？**
>     它加在 **“前向传播”** 的中间。具体来说，通常是在 **激活函数之后**，**下一层计算之前**。
>    - 流程：`Linear` -> `ReLU` -> `Dropout (这里置0)` -> `Next Linear`。

```python
import torch
import torch.nn as nn

def dm01():
    # 1. 随机创建输入数据
    t1 = torch.randint(0, 10, size = (1,4)).float()

    # 2. 创建隐藏层
    linear1 = nn.Linear(4, 5)
    l1 = linear1(t1)
    print(f'经过线性层加权求和后的数据：{l1}')

    # 3. 激活函数
    output = torch.relu(l1)
    print(f'失活前经过激活函数后的数据：{output}')

    # 4. 随机失活
    droupout = nn.Dropout(p = 0.4)  # 神经元有40%的概率被失活
    d1 = droupout(output)
    print(f'被失活后的数据：{d1}')
    # 没有被失活的数据会进行缩放，缩放比例为1 / (1-p)，目的是为了让测试阶段更快速

if __name__ == '__main__':
    dm01()
```

## 02.2 批量归一化BN

> * BN 层通常放在**卷积层/全连接层之后**，**激活函数之前**。即经典的 **Conv → BN → ReLU** 结构

![image-20260210185836902](F:\note\deep_learning\pytorch_learning\day04_学习率衰减方法.assets\image-20260210185836902.png)

> * Batch Normalization 就像是神经网络的**稳压器**。它通过强制调整每一层数据的均值和方差，让整个网络的梯度传播变得极其顺畅，允许你使用更大的学习率，并极大地加快了收敛速度

![image-20260210190156257](F:\note\deep_learning\pytorch_learning\day04_学习率衰减方法.assets\image-20260210190156257.png)

```python
# 批量归一化的思路：先对数据做标准化（会丢失一部分信息），然后再对数据做缩放（λ，理解为权重）和平移（β，理解为偏置），找补回一些丢失的信息

import torch
import torch.nn as nn

# 处理2维数据
def dm01():
    # 1. 创建图片样本数据
    input_2d = torch.randn(size = (1, 2, 3, 4))  # 1张2通道，3行4列（像素点）的图像

    # 2. 创建批量归一化层（BN层）
    bn2d = nn.BatchNorm2d(num_features=2, eps= 1e-5, momentum= 0.1, affine=True)  # 这个num_features,二维的就填通道数
    output_2d = bn2d(input_2d)
    print(f'output_2d:{output_2d}')

# 处理1维数据
def dm02():
    # 1. 创建样本数据
    input_1d = torch.randn(size=(2, 2))  # 2行2列的数据

    # 2. 创建线性层
    linear1 = nn.Linear(2, 4)
    l1 = linear1(input_1d)

    # 3. 创建BN层
    bn1d = nn.BatchNorm1d(num_features=4)  # 一维的填特征数
    output_1d = bn1d(l1)
    print(f'output_1d:{output_1d}')

if __name__ == '__main__':
    dm01()
    dm02()
```

> * 无论是 `BatchNorm1d` 还是 `BatchNorm2d`，它们的参数几乎是一样的。
>
>    ```python
>    nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>    ```
>
> * num_features是最关键的参数。BN 是对每个通道分别做归一化的。如果上一层是 `Conv2d(..., out_channels=64)`，这里必须填 `64`。 如果上一层是 `Linear(..., out_features=128)`，这里必须填 `128` (用 `BatchNorm1d`)。
>
> * AI给的示例：

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ============================
# 1. 准备数据 (分类任务)
# ============================
# 生成一个稍微难一点的分类数据集
n_samples = 2000
n_features = 50
n_classes = 3

torch.manual_seed(42)
X = torch.randn(n_samples, n_features)
# 人为制造不同特征维度的分布差异 (模拟 Covariate Shift 的前兆)
X[:, :25] = X[:, :25] * 5 + 10  # 前25个特征 均值10，方差25
X[:, 25:] = X[:, 25:] * 1 - 5   # 后25个特征 均值-5，方差1

# 生成标签
W_true = torch.randn(n_features, n_classes)
logits = X @ W_true
y = torch.argmax(logits, dim=1)

# 数据加载器
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# ============================
# 2. 定义两个模型
# ============================
class NetWithBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.BatchNorm1d(64), # 加在激活函数之前 (经典用法)
            nn.ReLU(),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Linear(32, n_classes)
        )
    def forward(self, x):
        return self.net(x)

class NetNoBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            # 没有 BN
            nn.ReLU(),
            
            nn.Linear(64, 32),
            # 没有 BN
            nn.ReLU(),
            
            nn.Linear(32, n_classes)
        )
    def forward(self, x):
        return self.net(x)

# ============================
# 3. 训练对比函数
# ============================
def train_model(model, name, lr=0.01):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    losses = []
    
    model.train() # 开启训练模式 (对 BN 至关重要)
    for epoch in range(20): # 训练 20 轮
        epoch_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))
    return losses

# ============================
# 4. 执行实验
# ============================
# 注意：我们可以尝试用较大的学习率，BN 通常能承受更大的 LR
lr = 0.05 

model_bn = NetWithBN()
losses_bn = train_model(model_bn, "With BN", lr=lr)

model_nobn = NetNoBN()
losses_nobn = train_model(model_nobn, "Without BN", lr=lr)

# ============================
# 5. 可视化
# ============================
plt.figure(figsize=(10, 5))
plt.plot(losses_bn, label='With Batch Norm', linewidth=2)
plt.plot(losses_nobn, label='Without Batch Norm', linestyle='--')
plt.title(f'Training Loss Comparison (LR={lr})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
```

![image-20260210195439883](F:\note\deep_learning\pytorch_learning\day04_学习率衰减方法.assets\image-20260210195439883.png)

![image-20260210195257706](F:\note\deep_learning\pytorch_learning\day04_学习率衰减方法.assets\image-20260210195257706.png)

