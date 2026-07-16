# 01 参数初始化

## 01.1 概念

> * 参数初始化的目的，用一句话概括就是：**给神经网络一个“健康的起跑线”，让它能跑得动（有梯度）、跑得快（收敛快），而且别跑偏（打破对称性）**
> * 这个对称性是指同一层的神经元的参数，打破对称性让每个神经元的初始状态不一样，从而可以独立的学习
> * 参数初始化位于下面所示的环节中：

1.**数据准备**：清洗、Dataset、DataLoader

2.**构建模型** (Model Construction) <--- **【参数初始化在这里！】**

- 定义层结构 (`__init__`)
- **覆盖默认参数 (`init.kaiming_...`)**
- 定义前向传播 (`forward`)

3.**设置Loss/优化器**：优化器接收已经初始化好的参数

4.**训练**：通过梯度下降，修改这些参数的值

5.**评估**

6.**可视化**

```python
import torch.nn as nn

def dm01():
    # 1.创建一个输入维度5，输出维度3的线性层
    linear = nn.Linear(5, 3) 
    # 2.对权重和偏置进行随机初始化，从0-1均匀分布产生
    nn.init.uniform_(linear.weight)  #带下划线_，表示原地操作，改变原始的值
    nn.init.uniform_(linear.bias)
    # 3.打印查看
    print(linear.weight.data)
    print(linear.bias.data)

def dm02():
    linear = nn.Linear(5, 4)
    # 1.固定初始化参数，这样无法打破对称性
    nn.init.constant_(linear.weight, 2)  # 固定为2
    nn.init.constant_(linear.bias, 1)    # 固定为1
    # 2.打印查看
    print(linear.weight.data)
    print(linear.bias.data)

def dm03():
    linear = nn.Linear(6, 4)
    # 1.全0参数初始化
    nn.init.zeros_(linear.weight)  
    nn.init.zeros_(linear.bias)
    # 2.打印查看
    print(linear.weight.data)
    print(linear.bias.data)
   
def dm04():
    linear = nn.Linear(6, 4)
    # 1.全1参数初始化
    nn.init.ones_(linear.weight)  
    nn.init.ones_(linear.bias)
    # 2.打印查看
    print(linear.weight.data)
    print(linear.bias.data)

def dm05():
    linear = nn.Linear(3, 2)
    # 1.标准正态分布参数初始化
    nn.init.normal_(linear.weight)  
    nn.init.normal_(linear.bias)
    # 也可以自定义，例如 nn.init.normal(mean = 2, std = 2)
    # 2.打印查看
    print(linear.weight.data)
    print(linear.bias.data)

def dm06():
    linear = nn.Linear(3, 2)
    # 1.kaiming正态分布初始化
    nn.init.kaiming_normal_(linear.weight) 
    # kaiming均匀分布 nn.init.kaiming_uniform_(linear.weight) 
    # 2.打印查看
    print(linear.weight.data)
    print(linear.bias.data) # bias不是kaiming初始化的，而是pytorch默认方式初始化的
    # 在实战中，偏置通常初始化为 0 就足够好了

def dm07():
    linear = nn.Linear(3, 2)
    # 1.xavier正态分布初始化
    nn.init.xavier_normal_(linear.weight)  
    # xavier均匀分布 nn.init.xavier_unform_(linear.weight)
    # 2.打印查看
    print(linear.weight.data)
    print(linear.bias.data)

if __name__ == '__main__':
    dm01()
    dm02()
    dm03()
    dm04()
    dm05()
    dm06()
    dm07()
```

> * 下面是AI给的示例：

 ```python
 import torch.nn as nn
 import torch.nn.init as init
 
 # ============================
 # 第 2 步：构建模型
 # ============================
 class MyNet(nn.Module):
     def __init__(self):
         super(MyNet, self).__init__()
         
         # A. 定义层 (PyTorch 在这里做了默认初始化)
         self.fc1 = nn.Linear(100, 50)
         self.relu = nn.ReLU()
         self.fc2 = nn.Linear(50, 10)
         
         # B. 手动初始化 
         # 我们在这里覆盖掉 PyTorch 的默认值
         self._init_weights() 
 
     def forward(self, x):
         x = self.relu(self.fc1(x))
         x = self.fc2(x)
         return x
 
     # 定义一个私有方法来专门做初始化
     def _init_weights(self):
         # 对 fc1 使用 Kaiming 初始化
         init.kaiming_normal_(self.fc1.weight)
         init.constant_(self.fc1.bias, 0)
         
         # 对 fc2 使用 Xavier 初始化
         init.xavier_normal_(self.fc2.weight)
         init.constant_(self.fc2.bias, 0)
 
 # 实例化模型 (此时初始化代码被执行！)
 model = MyNet() 
 ```

![image-20260210080406070](F:\note\deep_learning\pytorch_learning\day03_nn模块.assets\image-20260210080406070.png)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # === 场景 1: 标准骨干网络 (推荐 __init__) ===
        # 原因：结构清晰，支持算子融合，print(model) 时可见
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True) # 这里使用类的方式，清晰地定义了层结构
        )
        
        # === 场景 2: 有参数的激活函数 (必须 __init__) ===
        # 原因：PReLU 包含可学习的参数 'weight'，必须作为层注册到模型中
        # 如果只在 forward 里写 F.prelu(x, weight=???)，你需要手动管理参数，极其痛苦
        self.prelu = nn.PReLU() 
        
        # === 场景 3: Dropout 层 (强烈推荐 __init__) ===
        # 原因：nn.Dropout 会自动根据 model.train() 和 model.eval() 切换状态
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # --- 第一步：调用定义好的层 ---
        out = self.conv_block(x) 
        
        # --- 第二步：调用带参数的激活函数 ---
        out = self.prelu(out)
        
        # --- 第三步：使用函数式写法处理“魔改”逻辑 (推荐 forward) ---
        # 假设我们要实现一个类似 "Swish" 或 "Gating" 的操作： x * sigmoid(x)
        # 这种简单的数学运算，没必要专门在 init 里定义一个 nn.Sigmoid()
        # 直接调用 F.sigmoid 或者 torch.sigmoid 更符合直觉，代码也更短
        gate = torch.sigmoid(out) 
        out = out * gate
        
        # --- 第四步：残差连接中的激活 (通常用 forward) ---
        # 在 ResNet 的实现中，最后的相加和激活通常写在 forward 里
        # 这样可以看出它是在“运算”，而不是一个独立的“层”
        out = out + x  # 假设维度相同进行残差相加（仅作示意）
        out = F.relu(out) # 这里的 ReLU 仅仅是一个数学操作，不维护状态
        
        # --- 第五步：Dropout ---
        out = self.dropout(out) # 自动处理训练/测试模式
        
        return out

# 实例化模型
model = HybridNet()
```

## 01.2 选择方式

![image-20260206204643630](F:\note\deep_learning\pytorch_learning\day03_nn模块.assets\image-20260206204643630.png)

# 02 框架搭建流程

![image-20260207095214381](F:\note\deep_learning\pytorch_learning\day03_nn模块.assets\image-20260207095214381.png)

> * 例题：

![image-20260207095805507](F:\note\deep_learning\pytorch_learning\day03_nn模块.assets\image-20260207095805507.png)

```python
import torch
import torch.nn as nn
from torchsummary import summary  # 用于计算模型参数，查看模型结构

# 1.定义类，继承nn.Module
class ModelDmeo(nn.Module):
    # 1.1 在init魔法方法中完成初始化
    def __init__(self):
        # 初始化父类成员
        super().__init__()
        # 1.1.1 搭建神经网络，隐藏层和输出层
        # 隐藏层1：输入特征3，输出特征3
        self.linear1 = nn.Linear(3, 3)
        # 隐藏层2：输入3，输出2
        self.linear2 = nn.Linear(3, 2)
        # 输出层：输入2，输出2
        self.output = nn.Linear(2, 2)
        # 1.1.2 对隐藏层进行参数初始化
        # 隐藏层1
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        # 隐藏层2
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    # 1.2 定义前向传播函数,必须叫forward()
    def forward(self, x):
        # 第一层计算：加权求和+激活函数sigmoid
        x = self.linear1(x)
        x = torch.sigmoid(x)
        # 或者 x =torch.sigmod(self.linear1(x))
        # 第二层计算：加权求和+激活函数relu
        x = torch.relu(self.linear2(x))
        # 第三层计算：加权求和+激活函数softmax
        x = torch.softmax(self.output(x), dim = -1)  # dim = -1表示按行计算，一条样本一条样本的处理
        return x
    
# 2.模型训练
def train():
    # 2.1 创建模型对象
    my_model = ModelDmeo()
    print(f'my_model:{my_model}')
    # 2.2 创建数据集
    data = torch.randn(size=(5,3))
    # 2.3 调用模型进行训练
    output = my_model(data)
    print(f'output:{output}')
    print(f'output.shape:{output.shape}')
    # 2.4 检查模型结构
    summary(my_model, input_size=(5, 3)) # 第二个参数为输入数据的维度

if __name__ == '__main__':
    train() 
```

# 03 损失函数

> * **损失函数（Loss Function）**是衡量模型预测值与真实标签之间差异的“尺子”。训练过程的本质就是通过优化算法（如 SGD 或 Adam）不断减小损失函数的值
> * 多分类交叉熵损失函数使用时，输出层不需要再用softmax，因为交叉熵内部已经集成了

![image-20260207153147587](F:\note\deep_learning\pytorch_learning\day03_nn模块.assets\image-20260207153147587.png)

## 03.1 多分类任务：交叉熵损失

> * 标签必须是 **`LongTensor` (整数索引)**
> * 标签形状是 **一维 `(N,)`**
> * 模型输出 **不需要 Softmax**
> * 下面这个例子注意，用的是全批量梯度下降，而不是常见的小批量梯度下降，所以没有用到dataloader，循环也只有一层for循环

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. 准备数据
torch.manual_seed(42) # 固定随机种子
# 假设有 100 个样本，每个样本有 5 个特征
inputs = torch.randn(100, 5)
# 标签是 0, 1, 2 三类中的一个
# 注意：CrossEntropyLoss 的标签必须是 Long 类型 (int64)
targets = torch.randint(0, 3, (100,), dtype=torch.long)

# 2. 定义模型
# 输出维度必须等于类别数 (3)
model = nn.Linear(5, 3)

# 3. 定义 Loss 和 优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 4. 训练循环
print("--- 开始训练 CrossEntropyLoss ---")
for epoch in range(200):
    # a. 清零梯度
    optimizer.zero_grad()
    
    # b. 前向传播
    outputs = model(inputs) # 形状 (100, 3)
    
    # c. 计算 Loss
    loss = criterion(outputs, targets)
    
    # d. 反向传播
    loss.backward()
    
    # e. 更新参数
    optimizer.step()
    
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

# 简单验证：看看第一个样本预测对了没
with torch.no_grad():
    pred_score = model(inputs[0].unsqueeze(0))
    pred_cls = pred_score.argmax(dim=1).item()
    print(f"样本0 - 真实类别: {targets[0].item()}, 预测类别: {pred_cls}")
```

## 03.2 回归任务：均方误差

> * 标签必须是 **`Float`**
> * 标签形状必须和预测值一致，通常是 **二维 `(N, 1)`**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. 准备数据
# 生成 y = 2x + 1 的数据
X = torch.linspace(-10, 10, 100).reshape(100, 1) # (100, 1)
noise = torch.randn(100, 1) * 2
Y_true = 2 * X + 1 + noise # 加点噪声

# 2. 定义模型
model = nn.Linear(1, 1) # 输入1维，输出1维数值

# 3. 定义 Loss 和 优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. 训练循环
print("\n--- 开始训练 MSELoss ---")
for epoch in range(500):
    optimizer.zero_grad()
    
    predictions = model(X)
    
    # 注意：predictions 和 Y_true 形状必须完全一样 (100, 1)
    loss = criterion(predictions, Y_true)
    
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

# 验证参数
w, b = model.weight.item(), model.bias.item()
print(f"训练结果: w={w:.2f} (真实2.0), b={b:.2f} (真实1.0)")
```

## 03.3 稳健回归：平滑L1损失

> * 当数据里有**脏数据（离群点）**时使用。比如大部分房价是 200万，突然有个数据是 100亿（可能是写错了）。MSE 会被这个 100亿 吓死，拼命去拟合它，导致正常数据预测不准。SmoothL1 对这种异常值不敏感
> * 代码结构和 MSE 几乎一模一样，只是换了 Loss 函数

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. 准备数据
X = torch.linspace(-10, 10, 100).reshape(100, 1)
Y_true = 2 * X + 1
# 手动制造几个离群点 (Outliers)
Y_true[0] = 1000  # 搞个巨大的异常值
Y_true[1] = -1000 

# 2. 定义模型
model = nn.Linear(1, 1)

# 3. 定义 Loss 和 优化器
# beta=1.0 是默认参数，误差小于1用L2，大于1用L1
criterion = nn.SmoothL1Loss(beta=1.0) 
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. 训练循环
print("\n--- 开始训练 SmoothL1Loss ---")
for epoch in range(500):
    optimizer.zero_grad()
    predictions = model(X)
    loss = criterion(predictions, Y_true)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

print("SmoothL1 能较好地忽略那两个离群点，拟合正常数据。")
```

## 03.4 二分类任务：二元交叉熵

> * 标签必须是 **`Float`** (0.0 或 1.0)
> * 标签形状必须是 **二维 `(N, 1)`**
> * 模型输出 **不需要 Sigmoid** (因为 `WithLogits` 名字里带了 Logits，说明它内部会自己做 Sigmoid)

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. 准备数据
# 100 个样本，每个样本 10 个特征
inputs = torch.randn(100, 10)
# 标签是 0 或 1，但必须是 float 类型
targets = torch.randint(0, 2, (100, 1)).float()

# 2. 定义模型
# 二分类，最终输出 1 个数值 (代表 logit)
model = nn.Linear(10, 1)

# 3. 定义 Loss 和 优化器
# 强烈推荐用 BCEWithLogitsLoss 而不是 BCELoss，数值更稳定
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 4. 训练循环
print("\n--- 开始训练 BCEWithLogitsLoss ---")
for epoch in range(200):
    optimizer.zero_grad()
    
    # 输出的是 logits (实数范围)
    logits = model(inputs)
    
    # 计算 Loss
    loss = criterion(logits, targets)
    
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 50 == 0:
        # 计算准确率 (为了好看)
        # 先用 sigmoid 变回概率，再四舍五入(>0.5就是1)
        preds = torch.sigmoid(logits).round()
        acc = (preds == targets).float().mean()
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | Acc: {acc.item():.2f}")
```

# 04 网络优化方法

## 04.1 梯度下降算法

梯度下降是一种寻找使损失函数最小化的方法。梯度的方向是函数增长速度最快的方向，那么梯度的反方向就是函数减少最快的方向。

> * 在 PyTorch 中，所有的优化算法都封装在 `torch.optim` 模块里
> * epoch是使用全部数据对模型进行一次完整训练，训练的轮次。在一个epoch中，batch和iteration的值是相等的。

 **(1) Batch (批次) & Batch Size (批大小)**

- **定义**：模型一次能“吃”进去并进行计算的数据量。
- **比喻**：你不可能一口气把 1000 道题全做完再对答案（脑子记不住，显存不够）。你决定**每次做 10 道题**，然后对一次答案，改一次错。
   - 这里的 **10** 就是 **Batch Size**。
   - 这 **10 道题** 的组合就是一个 **Batch**。
- **为什么需要它？**
   - **内存/显存限制**：GPU 显存有限，塞不下整个数据集。
   - **速度与稳定的平衡**：一次学 1 个太慢且不稳定（SGD），一次学所有太吃内存（BGD）。Batch Size（如 32, 64, 128）是最佳平衡点（Mini-Batch）。

**(2) Iteration (迭代 / Step)**

- **定义**：模型更新一次参数的过程（前向传播 + 算 Loss + 反向传播 + 更新权重）。
- **比喻**：你做完这 10 道题，对完答案，发现自己错在哪，然后**修改脑子里的知识点**。这个“修改一次”的过程，就是 1 个 Iteration。
- **数量关系**：
   - 如果你有 1000 道题，每次做 10 道。
   - 你需要做 1000÷10=100 次才能把整本书过一遍。
   - 也就是 **1 个 Epoch = 100 个 Iterations**。

 **(3) Epoch (轮次)**

- **定义**：模型把**整个训练集**里的所有数据都“学”过了一遍。
- **比喻**：你把《五年高考三年模拟》从第一页做到最后一页，**完整地刷了一遍书**。这就叫 1 个 Epoch。
- **为什么需要多个 Epoch？**
   - 书看一遍能学会吗？肯定不行。通常需要**反复刷很多遍**（Epochs），才能彻底掌握书里的规律（收敛）。



**举个实战例子：**

- **数据**：你有一份包含 **10,000** 张图片的训练集。
- **设置**：你设置 `Batch Size = 100`，打算训练 **5** 个 Epochs。

**问：**

1. 每个 Epoch 有多少个 Iteration？
   - 10,000/100=100 个。
2. 整个训练过程一共更新了多少次参数（Total Iterations）？
   - 100 (每轮次数)×5 (轮数)=500 次。

![image-20260208154120658](F:\note\deep_learning\pytorch_learning\day03_nn模块.assets\image-20260208154120658.png)

```python
# 假设 dataset 有 1000 个样本
# batch_size = 10
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# epochs = 5 (整本书背 5 遍)
for epoch in range(5):  # ---【这里控制 Epoch】---
    
    # 每一轮开始
    # len(dataloader) = 100 (因为 1000 / 10 = 100)
    for i, (inputs, labels) in enumerate(dataloader): # ---【这里控制 Batch】---
        
        # ---【下面这 5 步叫 1 个 Iteration】---
        optimizer.zero_grad()       # 1. 清空
        outputs = model(inputs)     # 2. 前向 (做题)
        loss = criterion(outputs, labels) # 3. 算分 (对答案)
        loss.backward()             # 4. 求导 (找原因)
        optimizer.step()            # 5. 更新 (改错)
        
        # 打印日志通常是：
        # Epoch [1/5], Step [10/100], Loss: ...
```

### 04.1.1 随机梯度下降SGD

> * 如果把训练模型比作“下山找最低点”，那么**损失函数**决定了山的形状，**梯度**决定了下山的方向，而**优化器**决定了你**怎么走**（是小碎步走、跑着走、还是开着跑车走）
>
> * **原理**：看着当前的坡度（梯度），朝着反方向走一步
>
>    - 公式：$w_{new}=w_{old}−lr×gradient$
>
>    **特点**：
>
>    - **优点**：简单，内存占用极小
>    - **缺点**：
>       1. **怕峡谷**：在狭长的峡谷地形中，它会在两侧震荡，收敛极慢
>       2. **怕鞍点**：遇到平坦的地方（梯度接近0），它就走不动了

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

### 04.1.2 动量法Momentum

> * 为了解决 SGD “遇到平地走不动、遇到峡谷乱震荡”的问题，引入了物理学中的**惯性（Momentum）**
>
> * **原理**：**“滚雪球”**。
>
>    - 如果上一次向左更新了，这一次梯度还是向左，那就利用惯性加速，步子迈大点
>    - 如果上一次向左，这一次梯度突然向右（震荡），惯性会抵消一部分反向的力，走得更平稳
>
>    - **核心参数**：`momentum`（通常设为 **0.9**）
>    - **应用场景**：**计算机视觉（CV）的标准配置**。 ResNet、VGG 等经典论文通常都用 SGD + Momentum。虽然 Adam 收敛快，但 SGD+Momentum 往往能跑出**更高的最终准确率**（泛化能力强）

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

### 04.1.3 自适应法Adam

> * 这是目前最“智能”的派系。它们的核心思想是：**每个参数的学习率应该是独立的**。
>     （有的参数梯度大，经常更新，我们就让它学习率小点，稳一点；有的参数梯度小，很久不更新，我们就让它学习率大点，步子大点）。
>
>    #### 1. Adagrad
>
>    - **特点**：给常用参数小的学习率，给稀疏参数大的学习率。
>    - **缺点**：随着训练进行，学习率会不断衰减，最后直接变成 0，导致训练提前停止。
>    - **应用场景**：处理稀疏数据（如早期的 NLP 词向量任务），现在用得较少。
>
>    #### 2. RMSprop
>
>    - **特点**：改进了 Adagrad，解决了学习率过早消失的问题。它是 RNN（循环神经网络）时代的王者。
>    - **应用场景**：强化学习、RNN。
>
>    #### 3. Adam (Adaptive Moment Estimation) —— **无脑首选**
>
>    - **原理**：**Momentum + RMSprop 的结合体**。
>       - 既有动量（惯性），又能自适应调整每个参数的学习率。
>    - **特点**：
>       - **收敛极快**：通常是 SGD 的好几倍。
>       - **不挑食**：对学习率不敏感（默认 `1e-3` 或 `1e-4` 即可）。
>    - **应用场景**：**绝大多数任务的首选**（NLP、推荐系统、生成模型）。如果你不知道选谁，选 Adam 准没错。

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

![image-20260209211939176](F:\note\deep_learning\pytorch_learning\day03_nn模块.assets\image-20260209211939176.png)

```python
import torch
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 1. 制造一些假数据 (非线性回归)
torch.manual_seed(1)    # 随机种子
LR = 0.01               # 学习率
BATCH_SIZE = 32
EPOCH = 12

# y = x^2 + 0.1 * noise
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1 * torch.normal(torch.zeros(*x.size()))

# 放入 DataLoader
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 2. 定义一个简单的神经网络
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)   # 隐藏层
        self.predict = torch.nn.Linear(20, 1)  # 输出层

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

# 3. 创建 4 个完全一样的模型 (为了公平对比)
net_SGD         = Net()
net_Momentum    = Net()
net_RMSprop     = Net()
net_Adam        = Net()

# 4. 定义 4 种不同的优化器
# 注意：Momentum 也是用 SGD 调用的，只是加了 momentum 参数
opt_SGD         = torch.optim.SGD(net_SGD.parameters(), lr=LR)
opt_Momentum    = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
opt_RMSprop     = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
opt_Adam        = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))

# 把它们放入列表，方便循环
nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]
optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]
labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
loss_history = [[], [], [], []] # 记录 Loss 变化

# 5. 开始训练循环
print("开始对比训练...")
for epoch in range(EPOCH):
    print(f"Epoch: {epoch+1}")
    for step, (b_x, b_y) in enumerate(loader):
        
        # 对每个优化器进行一次更新
        for net, opt, l_his in zip(nets, optimizers, loss_history):
            output = net(b_x)              # 前向
            loss = F.mse_loss(output, b_y) # 算 Loss
            opt.zero_grad()                # 清零
            loss.backward()                # 反向
            opt.step()                     # 更新
            
            l_his.append(loss.item())      # 记录 Loss

# 6. 可视化结果
plt.figure(figsize=(10, 6))
for i, l_his in enumerate(loss_history):
    plt.plot(l_his, label=labels[i], alpha=0.7)

plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0, 0.2)) # 限制 y 轴范围，看清细节
plt.title('Different Optimizers Comparison')
plt.show()
```

![image-20260209212022141](F:\note\deep_learning\pytorch_learning\day03_nn模块.assets\image-20260209212022141.png)

### 04.1.4 指数移动加权EWMA

> * 深度学习中几乎所有高级优化算法（Momentum, RMSprop, Adam），本质上都是把“指数加权平均（EWMA）”这个数学工具，应用到了“梯度”或“梯度的平方”上

假设有一组连续的数据点 $\theta_1, \theta_2, \theta_3, ...$（比如最近 100 天的温度），这些数据可能因测量误差而上下跳动（**有噪声**）。想画一条平滑的曲线来表示温度的**趋势**。

**指数加权平均（Exponential Weighted Moving Average, EWMA）** 的公式如下：

$v_t=β⋅v_{t−1}+(1−β)⋅θ_t$

- $v_t$：当前时刻的“平滑值”（或者叫估计值）。
- $\theta_t$：当前时刻的“真实观测值”。
- $v_{t-1}$：上一时刻算出来的“平滑值”。
- $\beta$：**衰减率（超参数，0到1之间）**

- **β≈0.9**：
   - $v_t=0.9v_{t−1}+0.1θ_t$
   - 意思是：今天的趋势 = **90% 的历史遗产** + **10% 的今日新信息**。
   - **效果**：这相当于平均了过去约 $\frac{1}{1-\beta} $=10 天的数据。曲线很平滑，但反应有点迟钝（滞后）。
- **β≈0.98：**
   - 相当于平均了过去 $\frac{1}{0.02}$ =50 天的数据。曲线超级平滑，对当天的突变很不敏感。

**核心作用**：EWMA 像一个**滤波器**，它过滤掉了短期的高频噪声（Jitter），保留了长期的低频趋势（Trend）

## 04.2 反向传播

> * 简单来说：
>    - **反向传播**负责**“找原因”**（算出每个参数对错误的贡献有多少，即**计算梯度**）
>    - **梯度下降**负责**“改错误”**（根据找到的原因去更新参数，即**更新权重**）

假设你是一个学生（模型），正在参加一场考试。

1. **前向传播 (Forward Pass)**：
   - 你做完了题，交卷。
   - 老师批改，发现你得了 60 分（目标是 100 分）。
   - **Loss** = 40 分（误差）。
2. **反向传播 (Backpropagation) —— “找锅”**：
   - 为了提高成绩，你得知道**到底哪一步做错了**。
   - 你从**最后一道题**开始往前推：
      - “最后这道大题扣了 20 分，是因为中间那个公式背错了。” →\rightarrow→ **公式的梯度很大**（改它效果明显）。
      - “第一道填空题扣了 2 分，是因为粗心。” →\rightarrow→ **粗心的梯度很小**（改它提升空间不大）。
   - **核心**：反向传播就是从 Loss 出发，一层层向回推导，算出**每个神经元的权重（参数）对总误差有多大的责任**。这个“责任大小”就是**梯度**。
3. **梯度下降 (Gradient Descent) —— “改错”**：
   - 知道了哪一步错得最厉害，你开始复习（更新大脑里的知识权重）。
   - 公式错得最离谱，所以你花 80% 的精力背公式（大幅更新权重）。
   - 粗心影响不大，你花 5% 的精力练细心（微调权重）。

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ==========================================
# 1. 准备数据
# ==========================================
torch.manual_seed(42) # 固定随机种子，保证每次运行结果一样

# 生成 -3 到 3 之间的 100 个数
x = torch.linspace(-3, 3, 100).unsqueeze(1) # Shape: (100, 1)
# 真实标签 y = x^2
y = x.pow(2) + 0.1 * torch.randn(x.size())  # 加一点噪声

# ==========================================
# 2. 定义网络 (两层 MLP)
# ==========================================
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # 隐藏层：输入1 -> 输出10
        self.hidden = nn.Linear(1, 10)
        # 激活函数
        self.relu = nn.ReLU()
        # 输出层：输入10 -> 输出1
        self.predict = nn.Linear(10, 1)

    def forward(self, x):
        # 数据流向：x -> hidden -> relu -> predict -> out
        x = self.hidden(x)
        x = self.relu(x)
        x = self.predict(x)
        return x

model = SimpleNet()

# ==========================================
# 3. 定义损失函数与优化器
# ==========================================
criterion = nn.MSELoss()  # 均方误差
# SGD 就是梯度下降算法的实现者
# lr=0.1 表示学习率，即改错的幅度
optimizer = optim.SGD(model.parameters(), lr=0.1) 

print("训练前，查看一下隐藏层第一个权重的梯度：")
# 此时还没有反向传播，所以梯度应该是 None
print(f"model.hidden.weight.grad: {model.hidden.weight.grad}") 


# ==========================================
# 4. 训练循环 (Training Loop)
# ==========================================
print("\n开始训练...")
epochs = 500

for epoch in range(epochs):
    
    # --- Step 1: 前向传播 (Forward) ---
    # 也就是“做题”
    prediction = model(x)
    
    # --- Step 2: 计算损失 (Loss) ---
    # 也就是“批改打分”
    loss = criterion(prediction, y)
    
    # --- Step 3: 梯度清零 ---
    # 在算新账之前，把旧账本清空
    optimizer.zero_grad()
    
    # --- Step 4: 反向传播 (Backward) ---
    # 核心步骤！PyTorch 自动计算所有参数的梯度
    # 这行代码执行完后，所有 w.grad 里就有值了
    loss.backward()
    
    # --- Step 5: 梯度下降 (Optimization) ---
    # 核心步骤！根据 Step 4 算出的梯度，更新参数
    # w = w - lr * grad
    optimizer.step()
    
    # 每 100 轮打印一次
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ==========================================
# 5. 验证反向传播是否生效
# ==========================================
print("\n训练后，再次查看隐藏层第一个权重的梯度：")
# 此时 grad 里面应该有具体的数值了
print(f"model.hidden.weight.grad (部分): \n{model.hidden.weight.grad[0]}")


# ==========================================
# 6. 可视化结果
# ==========================================
plt.figure(figsize=(10, 6))
plt.scatter(x.data.numpy(), y.data.numpy(), label='Real Data', alpha=0.5)
plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=3, label='Predicted Curve')
plt.legend()
plt.title("Backpropagation & Gradient Descent Fitting x^2")
plt.show()
```

![image-20260209094134609](F:\note\deep_learning\pytorch_learning\day03_nn模块.assets\image-20260209094134609.png)





