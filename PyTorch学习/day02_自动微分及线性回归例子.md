# 01自动微分模块

> * 就是对损失函数求导（关于权重的导数），结合反向传播，更新权重参数

```python
import torch
import matplotlib.pyplot as plt

# --- 1. 准备数据 (Data) ---
# y = 3x + 0.8
# 生成 10 个 x 数据，形状 (10, 1)
X = torch.linspace(0, 10, 10).reshape(10, 1) 
# 生成对应的 y 数据，并加一点点噪声干扰
Y_true = 3 * X + 0.8 + torch.randn(10, 1) * 0.5 

# --- 2. 定义参数 (Parameters) ---
# 我们不知道真实的 w 是 3，b 是 0.8，我们要让机器猜出来。
# 随机初始化 w 和 b
# requires_grad=True 是必须的！否则无法求导
w = torch.randn(1, 1, requires_grad=True) 
b = torch.randn(1, 1, requires_grad=True)

print(f"初始猜测: w={w.item():.2f}, b={b.item():.2f}")

# --- 3. 设置超参数 ---
learning_rate = 0.05  # 学习率：每次迈多大步
epochs = 100          # 训练次数：学多少轮

# --- 4. 训练循环 (Training Loop) ---
for epoch in range(epochs):
    
    # -----------------------------------
    # A. 前向传播 
    # -----------------------------------
    # 矩阵乘法: (10,1) @ (1,1) -> (10,1)
    # 利用广播机制: (10,1) + (1,1) -> (10,1)
    Y_pred = X @ w + b 
    
    # -----------------------------------
    # B. 计算损失 
    # -----------------------------------
    # 1. 算出差值的平方: (Y_pred - Y_true)^2
    # 2. 求平均值 (mean)，把向量变成标量！
    # 注意：这里必须要 .mean()，否则 backward 无法对向量求导
    loss = (Y_pred - Y_true).pow(2).mean()
    
    # -----------------------------------
    # C. 反向传播 (求梯度)
    # -----------------------------------
    # 这一步算出了 loss 对 w 和 b 的导数
    # 结果存放在 w.grad 和 b.grad 中
    loss.backward()
    
    # -----------------------------------
    # D. 手动更新参数 (重点！)
    # -----------------------------------
    # 必须用 torch.no_grad() 包裹！
    # 意思：接下来的减法运算，不要记录到计算图里。
    # 如果不包，PyTorch 会以为这个减法也是模型的一部分，
    # 导致显存爆炸和逻辑错误。
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        
        # -----------------------------------
        # E. 梯度清零 (重点！)
        # -----------------------------------
        # 如果不清零，下一轮的梯度会加在这一轮的上面。
        # 这里使用 in-place 操作 zero_()
        w.grad.zero_()
        b.grad.zero_()
    
    # 每 10 轮打印一次进度
    if epoch % 10 == 0:
        print(f"轮次 {epoch} | Loss: {loss.item():.4f} | w: {w.item():.2f}, b: {b.item():.2f}")

# --- 5. 最终结果 ---
print("\n训练结束！")
print(f"真实值: w=3.00, b=0.80")
print(f"预测值: w={w.item():.2f}, b={b.item():.2f}")
```

# 02 线性回归例子

> * 在用pytorch做深度回归时，需要把numpy对象转成张量tensor，然后再转成数据集对象TensorDataset转成数据加载器DataLoader
> * 这里其实张量相当于是最原始的食材，然后TensorDataset相当于是打包好的饭盒，DataLoader是传送带，负责一次批量拿几个饭盒
> * Tensor (存数据) → TensorDataset (对齐数据) → DataLoader (加载数据)→ Model

```python
import torch
from torch.utils.data import TensorDataset   #构造数据集对象
from torch.utils.data import DataLoader   # 数据加载器
from torch import nn     # nn模块中有均方误损失函数和假设函数
from torch import optim    # optim模块中有优化器函数
from sklearn.datasets import make_regression   # 创建线性回归模型数据集
import matplotlib.pyplot as plt   

plt.rcParams['font.sans-serif'] = ['SimHei']   # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

# 1.生成数据集
def create_datasets():
    # 1.1创建数据集对象
    x, y, coef = make_regression(n_samples=100,
                                 n_features=1,
                                 noise = 10,
                                 coef = True,
                                 random_state = 42
                                 )
    print(type(x))  # 这里的x还是ndarray对象
    # 1.2转成tensor张量，后面只用x和y，coef不需要
    x = torch.tensor(x, dtype = torch.float)
    y = torch.tensor(y, dtype = torch.float)

    return x, y, coef


# 2.创建训练模型
def train(x, y, coef):
    # 2.1 创建数据集对象
    dataset = TensorDataset(x, y)
    # 2.2 创建数据集加载器对象
    dataloader = DataLoader(dataset, batch_size = 16, shuffle = True)  
    # 2.3 创建线性回归模型
    model = nn.Linear(1,1)  # 输入1个特征，输出1个标签
    # 2.4 创建损失函数对象
    criterion = nn.MSELoss()
    # 2.5 创建优化器对象
    optimizer = optim.SGD(model.parameters(), lr = 0.01)
    # 2.6 定义训练流程
    epochs, loss_list, total_loss, total_sample = 100, [], 0.0, 0
    for epoch in range(epochs):
        # 每轮都是分批次训练的，所以从dataloader里获取批次数据
        for train_x, train_y in dataloader:
            # 前向计算
            y_pred = model(train_x)
            # 计算每批的平均损失，指定y要1列，-1是自动计算行数
            loss = criterion(y_pred, train_y.reshape(-1, 1))  
            # 计算总损失和批次数
            total_loss += loss.item()
            total_sample += 1
            # 梯度清零
            optimizer.zero_grad()
            # 计算梯度，方向传播
            loss.backward()
            # 更新梯度
            optimizer.step()
        loss_list.append(total_loss / total_sample)
        print(f'轮次：{epoch + 1}, 平均损失：{total_loss / total_sample}')

    # 2.7 打印最终训练结果
    print(f'模型学习的参数，权重为{model.weight},偏置为{model.bias}')

    # 2.8结果可视化
    plt.plot(range(epochs), loss_list)
    plt.title('损失曲线图')
    plt.grid()
    plt.show()

    plt.scatter(x, y, color = 'red', label = '真实值')
    y_pred = torch.tensor(data = [v * model.weight + model.bias for v in x])
    plt.plot(x, y_pred, color = 'blue', label = '预测曲线')
    plt.title('真实值vs拟合曲线')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    x, y, coef = create_datasets()
    train(x, y, coef)
```

> * 例子二

```python
import torch
import torch.nn as nn                  # 神经网络工具箱
import torch.optim as optim            # 优化器工具箱
from torch.utils.data import TensorDataset, DataLoader # 数据处理工具
import matplotlib.pyplot as plt        # 画图工具

# ==========================================
# 第一步：准备数据集 (Data Preparation)
# ==========================================
# 1. 生成一些假数据
# 我们生成 100 个 x，范围在 -10 到 10 之间
# view(-1, 1) 是把形状从 (100) 变成 (100, 1)，这是 nn.Linear 要求的输入格式(矩阵)
true_w = 3
true_b = 0.8
num_samples = 100

X = torch.linspace(-10, 10, num_samples).view(-1, 1)
# 生成对应的 y，并加上一些随机噪声 (模拟真实世界的误差)
Y = true_w * X + true_b + torch.randn(X.size()) * 2

# 2. 制作数据管道 (DataLoader)
# TensorDataset: 把 x 和 y 打包在一起，像拉链一样一一对应
dataset = TensorDataset(X, Y)

# DataLoader: 它是数据的搬运工
# batch_size=16: 每次搬运 16 个数据去训练 (小批量梯度下降)
# shuffle=True:  每次训练前把数据打乱，防止模型死记硬背顺序
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

print(f"数据准备完毕: 共 {len(dataset)} 条数据")


# ==========================================
# 第二步：构建模型 (Build Model)
# ==========================================
# 我们定义一个最简单的神经网络类
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        # 定义层：nn.Linear 表示全连接层 (线性层)
        # 1, 1 表示：输入特征是 1 个 (x)，输出特征也是 1 个 (y)
        # 它内部会自动初始化权重 w 和偏置 b，并开启梯度追踪
        self.linear = nn.Linear(1, 1) 

    def forward(self, x):
        # 前向传播：定义数据怎么流过这个网络
        # 输入 x -> 经过线性层 -> 输出 y_pred
        out = self.linear(x)
        return out

model = LinearRegressionModel()
print("模型构建完毕:", model)


# ==========================================
# 第三步：设置损失函数和优化器 (Loss & Optimizer)
# ==========================================
# 1. 损失函数：均方误差 (MSE)
# 它会自动计算 (预测值 - 真实值)^2 并求平均
criterion = nn.MSELoss()

# 2. 优化器：随机梯度下降 (SGD)
# model.parameters() 告诉优化器：“你要负责更新这个模型里的 w 和 b”
# lr=0.01 是学习率，控制每次更新的步子大小
optimizer = optim.SGD(model.parameters(), lr=0.01)


# ==========================================
# 第四步：模型训练 (Training Loop)
# ==========================================
print("\n开始训练...")
num_epochs = 100  # 把所有数据看 100 遍

for epoch in range(num_epochs):
    # 每一轮，我们从 DataLoader 里一车一车地拉数据
    for batch_x, batch_y in dataloader:
        
        # --- A. 清空过往梯度 ---
        # 必须清零！否则梯度会累加，导致更新出错
        optimizer.zero_grad()
        
        # --- B. 前向传播 (预测) ---
        # 现在的参数预测出来的结果
        y_pred = model(batch_x)
        
        # --- C. 计算损失 (打分) ---
        loss = criterion(y_pred, batch_y)
        
        # --- D. 反向传播 (求导) ---
        # PyTorch 自动算出 w 和 b 的梯度
        loss.backward()
        
        # --- E. 更新参数 ---
        # 优化器根据梯度，自动调整 w 和 b
        optimizer.step()
        
    # 每 20 轮打印一次当前的 Loss
    if (epoch+1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


# ==========================================
# 第五步：评估 (Evaluation)
# ==========================================
# 切换到评估模式 (对于简单线性回归这句不是必须的，但好习惯要养成)
model.eval()

# 停止梯度追踪 (省内存，因为不需要训练了)
with torch.no_grad():
    # 用训练好的模型预测所有 X
    predicted = model(X)
    
    # 取出训练好的参数看看
    # model.linear.weight 是一个张量，用 .item() 取出数值
    trained_w = model.linear.weight.item()
    trained_b = model.linear.bias.item()
    
    print("\n训练结束！")
    print(f"真实参数: w={true_w}, b={true_b}")
    print(f"训练参数: w={trained_w:.2f}, b={trained_b:.2f}")


# ==========================================
# 第六步：可视化 (Visualization)
# ==========================================
plt.figure(figsize=(10, 6))

# 画出真实的散点数据
# X.numpy() 把 tensor 转成 numpy 数组，因为 matplotlib 不认 tensor
plt.scatter(X.numpy(), Y.numpy(), label='Original Data', color='blue', alpha=0.5)

# 画出模型拟合的直线
plt.plot(X.numpy(), predicted.numpy(), label='Fitted Line', color='red', linewidth=2)

plt.legend()
plt.title("Linear Regression with PyTorch")
plt.show()
```

# 03 深度学习代码框架

第一步：准备数据集 (Data Preparation)

**【比喻】准备复习资料**

没有资料，学生没法学。

- **核心动作**：
   1. **清洗/转换**：把图片、文本变成 Tensor（`transforms.ToTensor`）。
   2. **Dataset**：把题目（x）和答案（y）装订成册（`TensorDataset`）。
   3. **DataLoader**：把书分成一章一章（Batch），而且要打乱顺序防止死记硬背（`shuffle=True`）。

> **代码关键词**：`TensorDataset`, `DataLoader`, `batch_size`, `shuffle`

------

第二步：构建模型 (Model Construction)

**【比喻】塑造学生的大脑**

这一步是决定这个学生够不够聪明（脑容量多大）。

- **核心动作**：
   1. **继承 `nn.Module`**：这是规矩。
   2. **`__init__` (买零件)**：定义你需要用到哪些层（全连接层 `Linear`、卷积层 `Conv2d`、激活函数 `ReLU`）。
   3. **`forward` (组装)**：定义数据怎么流过这些层。**注意**：如果是图片进全连接层，记得在这里把数据“拍扁” (`view` / `flatten`)。

> **代码关键词**：`class Net(nn.Module)`, `super().__init__()`, `self.layer`, `forward(x)`

------

第三步：设置损失函数与优化器 (Loss & Optimizer)

**【比喻】制定评分标准与教学方法**

- **损失函数 (Loss)**：**评分标准**。
   - 做填空题（回归）用 `MSELoss`（离得越近越好）。
   - 做选择题（分类）用 `CrossEntropyLoss`（选对类别才行）。
- **优化器 (Optimizer)**：**纠错老师**。
   - 它根据错题情况，告诉模型怎么修改参数。常用 `SGD` 或 `Adam`。
   - **学习率 (lr)**：纠错的力度。太大容易改过头，太小改得慢。

> **代码关键词**：`criterion = nn.MSELoss()`, `optimizer = optim.SGD(model.parameters(), lr=0.01)`



 第四步：模型训练 (Training Loop) —— **最容易晕的地方**

**【比喻】开始刷题（题海战术）**

这是核心中的核心，通常是一个双层循环：外层是 `Epoch`（把整本书看几遍），内层是 `Batch`（每次做几页题）。

**内层循环必须死记硬背的“五步走”：**

1. **`optimizer.zero_grad()`** —— **清空脑袋**。
   - *（忘掉上一题的解题思路，否则会干扰这一题）*。
2. **`y_pred = model(x)`** —— **做题**。
   - *（前向传播，算出预测结果）*。
3. **`loss = criterion(y_pred, y)`** —— **对答案**。
   - *（算出得了多少分，或者说错得有多离谱）*。
4. **`loss.backward()`** —— **找原因**。
   - *（反向传播，算出每个知识点（参数）该怎么改）*。
5. **`optimizer.step()`** —— **改正**。
   - *（更新参数，真正地修改大脑里的知识）*。

------

第五步：模型评估 (Evaluation)

**【比喻】模拟考试**

学得怎么样，要拿一套没做过的题（测试集）来考一考。

- **核心动作**：
   1. **`model.eval()`**：告诉模型“现在是考试模式”，关闭 Dropout 等只有训练时才用的功能。
   2. **`with torch.no_grad()`**：告诉 PyTorch“不用算梯度了”，省点脑子（显存），反正考试只看分数不改错。
   3. **统计准确率**：看看猜对了多少个。

------

第六步：可视化/预测 (Visualization)

**【比喻】成绩单与成果展示**

- **核心动作**：
   - 把 Loss 曲线画出来，看看学习过程是不是越来越好。
   - 挑几张图进去，看看模型预测出来是啥。
   - **注意**：Matplotlib 不认识 Tensor，画图前记得把 Tensor 转回 CPU 并变回 NumPy (`x.cpu().numpy()`)。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision            # 专门处理图像的库 (本次新增)
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ==========================================
# 第一步：准备数据集 (Data Preparation)
# ==========================================
# 我们不需要自己造假数据了，直接下载著名的 MNIST 数据集

# 定义一个转换器：把原始图片 (0-255的像素值) 转成 PyTorch 的 Tensor (0-1之间)
# 类似于把食材洗干净切好
transform = transforms.Compose([transforms.ToTensor()])

# 下载训练集
# root='./data': 下载到当前文件夹下的 data 目录
# train=True: 下载训练集 (60000张)
# download=True: 如果没下载过，就自动下载
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transform,  
                                           download=True)

# 下载测试集 (10000张，用来最后考试)
test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transform)

# 放入传送带 (DataLoader)
# batch_size=64: 一次拿 64 张图去训练
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

print(f"训练集数量: {len(train_dataset)}")
print(f"测试集数量: {len(test_dataset)}")


# ==========================================
# 第二步：构建模型 (Model Definition) —— 重点详解面向对象
# ==========================================
# class 就像是在画一张“图纸”。
# nn.Module 是 PyTorch 提供的标准模板，所有的神经网络都要照着这个模板画。
class DigitNet(nn.Module):
    
    # __init__ 是“初始化函数”。
    # 你可以把它理解为：买零件的清单。
    # 在这里，我们把网络需要的层（Layer）都定义好。
    def __init__(self):
        # 这一行是标准写法，必须写。意思是：“先运行父类(nn.Module)的初始化”
        super(DigitNet, self).__init__()
        
        # 定义第1层：全连接层
        # 输入: 784 (因为图片是 28x28 像素，拉平后是 784 个点)
        # 输出: 128 (隐藏层神经元数量，这个数字可以自己定，通常大一点效果好)
        # self.xx 意思就是把这个层变成了这个模型的一个“零件”
        self.fc1 = nn.Linear(in_features=784, out_features=128)
        
        # 定义激活函数：ReLU (把负数变成0，增加非线性)
        self.relu = nn.ReLU()
        
        # 定义第2层：输出层
        # 输入: 128 (承接上一层的输出)
        # 输出: 10 (代表 0-9 这10个数字的评分)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    # forward 是“前向传播函数”。
    # 你可以把它理解为：组装流水线。
    # 也就是数据 x 进来后，按什么顺序经过那些零件。
    def forward(self, x):
        # x 的原始形状是 (Batch, 1, 28, 28) -> 一摞二维图片
        # 1. 变形 (Reshape/Flatten)
        # 也就是把图片“拍扁”成一维向量，以便放入全连接层
        # -1 表示自动算 Batch Size，784 是 28*28
        x = x.view(-1, 784) 
        
        # 2. 经过第1层
        x = self.fc1(x)
        
        # 3. 经过激活函数
        x = self.relu(x)
        
        # 4. 经过第2层 (输出层)
        out = self.fc2(x)
        
        # 注意：这里不需要加 Softmax！因为后面的 Loss 函数自带了。
        return out

# 拿着图纸，造出一个真正的模型实例
model = DigitNet()
print(model)


# ==========================================
# 第三步：设置损失函数和优化器
# ==========================================
# 1. 损失函数：交叉熵损失 (CrossEntropyLoss)
# 它是分类任务的标配。它内部会自动做 Softmax (把评分变成概率) + Log + NLLLoss。
criterion = nn.CrossEntropyLoss()

# 2. 优化器：SGD
optimizer = optim.SGD(model.parameters(), lr=0.01)


# ==========================================
# 第四步：模型训练 (Training Loop)
# ==========================================
print("\n开始训练...")
num_epochs = 5  # 训练 5 轮 (MNIST比较简单，5轮够了)

for epoch in range(num_epochs):
    # i 是当前是第几批数据
    # images 是图片数据，labels 是对应的数字 (0-9)
    for i, (images, labels) in enumerate(train_loader):
        
        # 1. 清空梯度
        optimizer.zero_grad()
        
        # 2. 前向传播
        outputs = model(images)
        
        # 3. 计算损失
        # outputs 是 10 个评分，labels 是真实数字
        loss = criterion(outputs, labels)
        
        # 4. 反向传播
        loss.backward()
        
        # 5. 更新参数
        optimizer.step()
        
        # 每训练 300 批打印一次，防止屏幕刷屏太快
        if (i+1) % 300 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')


# ==========================================
# 第五步：模型评估 (Evaluation)
# ==========================================
print("\n开始考试 (测试集评估)...")
model.eval()  # 切换到评估模式

# 统计一共猜对了好个
correct = 0
total = 0

with torch.no_grad(): # 考试时不需要算梯度
    for images, labels in test_loader:
        # 1. 预测
        outputs = model(images)
        
        # 2. 获取预测结果
        # outputs 的形状是 (64, 10)，代表64张图，每张图有10个分值
        # torch.max(outputs, 1) 返回两个值：(最大值, 最大值的索引)
        # 我们只关心索引（predicted），因为索引就是数字本身（索引0代表数字0）
        _, predicted = torch.max(outputs.data, 1)
        
        # 3. 统计
        total += labels.size(0) # 这批一共有多少个
        correct += (predicted == labels).sum().item() # 猜对了多少个

accuracy = 100 * correct / total
print(f'测试集准确率: {accuracy:.2f}%')


# ==========================================
# 第六步：可视化 (Visualization)
# ==========================================
# 拿几张测试集的图出来看看效果
examples = iter(test_loader)
example_data, example_targets = next(examples)

with torch.no_grad():
    output = model(example_data)

# 画出前 6 张图
plt.figure(figsize=(10, 4))
for i in range(6):
    plt.subplot(1, 6, i+1)
    # 把 Tensor 转回 numpy 才能画图
    # [0] 是因为数据形状是 (1, 28, 28)，要去掉那个通道维度
    plt.imshow(example_data[i][0], cmap='gray') 
    
    # 获取模型预测的数字
    # output[i] 是一个长度为10的向量，argmax取出最大值的索引
    pred = output[i].argmax().item()
    
    plt.title(f"Pred: {pred}")
    plt.axis('off')

plt.show()
```









