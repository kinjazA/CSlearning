# 01 ANN完整案例

![image-20260210200347539](F:\note\deep_learning\pytorch_learning\day05_ANN案例.assets\image-20260210200347539.png)

```python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim 
from sklearn.model_selection import train_test_split
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

# todo 1. 定义函数，构建数据集
def create_dataset():
    # 1. 加载csv文件数据集
    data = pd.read_csv('手机价格预测.csv')
    print(f'data:{data.shape}')  # 2000行，20个特征，1个标签
    # 2. 获取特征x列和标签y列
    x , y = data.iloc[:,:-1] , data.iloc[:, -1]
    # 3. 把特征转成浮点型，便于后续自动微分
    x = x.astype(np.float32)
    # 4. 切分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)  #stratify参数是参考y的类别进行划分，更加均匀
    # 5. 封装成张量
    train_dataset = TensorDataset(torch.tensor(x_train.values), torch.tensor(y_train.values))
    test_dataset = TensorDataset(torch.tensor(x_test.values), torch.tensor(y_test.values))
    return train_dataset, test_dataset,x_train.shape[1], len(np.unique(y))


# todo 2. 搭建网络
class PhonePriceModel(nn.Module):
    # 1. 在init方法里初始化父类成员，搭建神经网络
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, 256)
        self.output = nn.Linear(256, output_dim)

    # 2. 定义forward函数，串好网络顺序
    def forward(self, x):
        # 2.1 第一层
        x = torch.relu(self.linear1(x))
        # 2.2 第二层
        x = torch.relu(self.linear2(x))
        # 2.3 第三层
        x = self.output(x)  # 因后面用的多分类交叉熵损失函数，这里不再用softmax
        return x


# todo 3. 模型训练
def train(model, train_dataset, input_dim, output_dim):
    # 1. 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # 2. 定义损失函数，多分类交叉熵损失
    criterion = nn.CrossEntropyLoss()
    # 3. 创建优化器对象
    optimizer = optim.Adam(model.parameters(), lr = 0.001, betas=(0.9, 0.99))
    # 4. 训练循环
    epochs = 50
    for epoch in range(epochs):
        # 定义变量，记录每次训练的损失值，训练批次数
        total_loss, batch_num = 0.0 , 0
        start = time.time()
        # 循环
        for x, y in train_loader:
            # 切换模型到训练状态
            model.train()
            # 4.1 模型预测，前向传播
            y_pred = model(x)
            # 4.2 计算loss
            loss = criterion(y_pred, y)
            # 4.3 梯度清零，反向传播，更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 累加损失值
            total_loss += loss.item()
        # 打印本轮信息
        print(f'epoch:{epoch+1}, loss:{total_loss / batch_num:.4f}, time:{time.time() - start:.2f}秒')
    # 训练结束，保存模型（参数）
    torch.save(model.state_dict(), './model/phone.pth')  # 后缀用pth,pkl,pickle均可


# todo 4. 模型测试
def evaluate(test_dataset, input_dim, output_dim):
    # 1. 创建神经网络对象
    model = PhonePriceModel(input_dim, output_dim)
    # 2. 加载模型参数
    model.load_state_dict(torch.load('./model/phone.pth'))
    # 3. 创建测试集数据加载器
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)  # 测试不需要打乱数据
    # 4. 定义变量，记录预测正确的个数
    correct = 0
    # 5. 按批次测试验证
    for x, y in test_loader:
        # 模型切换到测试状态
        model.eval()
        # 获取预测结果
        y_pred = model(x)
        y_pred = torch.argmax(y_pred, dim = 1)
        print(y_pred == y)
        print((y_pred == y).sum())
        correct += (y_pred == y).sum().item() 
    # 6. 打印准确率
    print(f'准确率为：{correct / len(test_dataset):.4f}')

     
if __name__ == '__main__':
    train_dataset, test_dataset, input_dim, output_dim = create_dataset()
    model = PhonePriceModel(input_dim, output_dim)
    summary(model, input_size=(16, input_dim))  # 每批16条，每条20列特征
    train(model, train_dataset, input_dim, output_dim)
    evaluate(test_dataset, input_dim, output_dim)
```

# 02 细节梳理

> * 为什么训练和测试两个函数里模型状态要切换？

代码里出现的 `model.train()` 和 `model.eval()` 就像是给模型穿“工作服”和“礼服”。

虽然在这个简单的 MLP模型里，这两句代码**实际上不起作用**（因为你没用特殊的层），但在工业界标准代码中，这是**必须写**的防御性代码。

- **`model.train()` (训练模式)**：
   - 告诉模型：“我们要开始学习了！”
   - **启用 Dropout 层**：随机让神经元失活（防止过拟合）。
   - **启用 BatchNorm 层**：使用当前这批数据（Batch）的均值和方差来做归一化，并更新全局统计量。
- **`model.eval()` (评估/测试模式)**：
   - 告诉模型：“我们要去考试了，别再学了，拿出你的固定水平！”
   - **停用 Dropout 层**：所有神经元全功率工作。
   - **锁定 BatchNorm 层**：不再计算当前数据的均值方差，而是使用训练时存下来的全局统计量。

**结论**：即使现在的模型里没有 Dropout 和 BN 层，养成写这两句的习惯至关重要。万一哪天往模型里加了一个 Dropout，却忘了写 `model.eval()`，测试结果就会一塌糊涂。

![image-20260211110720238](F:\note\deep_learning\pytorch_learning\day05_ANN案例.assets\image-20260211110720238.png)

# 03 AI完善后的

```python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim 
from sklearn.model_selection import train_test_split
import numpy as np
import time
import os

# ==========================================
# 1. 数据准备 (Data Preparation)
# ==========================================
def create_dataset():
    # --- 模拟生成数据 (为了让代码能直接运行) ---
    # 假设有 2000 个样本，20 个特征
    # 标签是 0, 1, 2, 3 (4个价格档位)
    np.random.seed(42)
    X = np.random.randn(2000, 20).astype(np.float32) # 特征必须是 float32
    y = np.random.randint(0, 4, size=(2000))         # 标签必须是 int
    
    # --- 划分训练集和测试集 ---
    # stratify=y 保证训练集和测试集里各类别的比例一致
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- 转换成 PyTorch 的 Tensor ---
    # 注意：特征是 float32，标签必须是 long (int64) 才能用于 CrossEntropyLoss
    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train).long())
    test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test).long())
    
    # 返回数据集和维度信息
    return train_dataset, test_dataset, x_train.shape[1], len(np.unique(y))

# ==========================================
# 2. 模型定义 (Model Definition)
# ==========================================
class PhonePriceModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),              # 激活函数
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim) 
            # 注意：最后一层不加 Softmax，因为 CrossEntropyLoss 内部自带了 Softmax
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. 训练函数 (Training Loop)
# ==========================================
# 【关键修改】把 model 作为参数传进来，不要在内部创建！
def train(model, train_dataset, epochs=100):
    # 1. 创建加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # 2. 定义损失和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(">>> 开始训练...")
    
    # 3. Epoch 循环
    for epoch in range(epochs):
        # 【最佳实践】每个 Epoch 开始时切换一次状态即可
        model.train() 
        
        running_loss = 0.0 # 记录这一轮的总 Loss
        start_time = time.time()
        
        # 4. Batch 循环 (核心逻辑)
        for x, y in train_loader:
            # --- 标准五步法 ---
            # 1. 预测
            y_pred = model(x)
            # 2. 算 Loss
            loss = criterion(y_pred, y)
            # 3. 清空梯度
            optimizer.zero_grad()
            # 4. 反向传播
            loss.backward()
            # 5. 更新参数
            optimizer.step()
            
            # --- 累加 Loss ---
            # loss.item() 是把 tensor 转成 float，防止显存爆炸
            running_loss += loss.item()
            
        # 5. 打印本轮信息
        # len(train_loader) 就是这一轮有多少个批次
        avg_loss = running_loss / len(train_loader) 
        print(f'Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Time: {time.time() - start_time:.2f}s')

    # 6. 保存模型
    if not os.path.exists('./model'):
        os.makedirs('./model') # 如果文件夹不存在，创建它
    torch.save(model.state_dict(), './model/phone.pth')
    print("模型已保存到 ./model/phone.pth")

# ==========================================
# 4. 评估函数 (Evaluation Loop)
# ==========================================
def evaluate(model, test_dataset):
    # 加载参数（确保我们评估的是训练好的参数）
    # 注意：实际项目中，如果是训练完直接评估，其实不需要重新 load，因为 model 已经在内存里更新了
    # 这里演示如何加载文件
    if os.path.exists('./model/phone.pth'):
        model.load_state_dict(torch.load('./model/phone.pth'))
        print(">>> 成功加载模型参数，开始评估...")
    else:
        print(">>> 未找到模型文件，使用当前内存中的模型评估...")

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 【最佳实践】切换到评估模式
    model.eval()
    
    correct = 0
    total = 0
    
    # 【最佳实践】评估不需要算梯度，一定要加 no_grad，省内存快一倍
    with torch.no_grad():
        for x, y in test_loader:
            outputs = model(x)
            
            # 获取预测类别：argmax(dim=1) 找概率最大的索引
            _, predicted = torch.max(outputs, 1)
            
            # 统计
            total += y.size(0) #这一批有多少个样本
            correct += (predicted == y).sum().item() # 猜对了多少个
            
    accuracy = 100 * correct / total
    print(f'测试集准确率: {accuracy:.2f}%')

# ==========================================
# 5. 主程序 (Main)
# ==========================================
if __name__ == '__main__':
    # 1. 获取数据
    train_data, test_data, input_n, output_n = create_dataset()
    print(f"输入特征数: {input_n}, 输出类别数: {output_n}")
    
    # 2. 实例化模型 (只在这里创建一次！)
    model = PhonePriceModel(input_n, output_n)
    
    # 3. 训练 (把模型传进去)
    train(model, train_data, epochs=100) 
    
    # 4. 评估 (把同一个模型传进去)
    evaluate(model, test_data)
```









