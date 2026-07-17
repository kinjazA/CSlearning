# 第5章：PyTorch 集成实战

## 本章目标
掌握在 PyTorch 训练循环中集成 Optuna 的完整流程，包括模型定义、训练、验证、保存最佳模型。

---

## 5.0 先问一句：你的场景真的需要上 Optuna 吗？

在动手之前，先做一个快速判断。不同场景下 Optuna 的投入产出比差异很大：

| 你的场景 | 值得用 Optuna？ | 理由 |
|---------|:---:|------|
| 自己设计的小网络（MLP/小型 CNN），结构不确定 | ✅ 非常值得 | 层数、宽度、激活函数都未知，搜索空间大，收益明显 |
| 标准架构（ResNet-50 / BERT），只调学习率 | ⚠️ 可选 | 社区已摸透默认参数，lr=1e-3 + Adam 经常是局部最优。花 10 次 trial 微调 lr 和 weight_decay 即可，100 次可能浪费 |
| Kaggle 比赛，要榨干最后 0.5% | ✅ 值得 | 每个 trial 跑短一点（剪枝）+ 重点搜 lr、augmentation、scheduler |
| 大模型微调（LLaMA/Qwen 等 LoRA） | ❌ 通常不值 | 全量 HPO 太贵。手动或简单网格搜 lr、rank、alpha 三个值就行 |
| 课程作业 / 快速原型 | ❌ 不值 | 用框架自带的 ReduceLROnPlateau + EarlyStopping 够用，Optuna 是过度工程 |

**一句话原则**：
> 你的模型架构越**不标准**、单次训练越**快**、超参数组合越**不确定**——Optuna 的收益越大。反过来，如果架构已经固定、训练一次要几小时、你能猜到大概的好参数范围——手动调或简单网格搜索就够了。

**和传统 ML 的区别**：
| 维度 | 传统 ML（sklearn/XGBoost） | 深度学习 |
|---|---|---|
| 单次训练耗时 | 秒~分钟 | 分钟~小时 |
| 能跑的 trial 数 | 100~1000 | 10~50 |
| 超参不确定性 | 高（模型类型本身也是搜索对象） | 低（架构通常已固定） |
| 边际收益 | **大**——好参数 vs 默认参数差异显著 | 中——默认参数通常已经不错 |
| 社区共识 | Optuna/Tune 几乎是标配 | Optuna 有一席之地，但不必须 |

**如果你决定用，DL 场景的建议配置**：
```python
# DL 场景通常 trial 数量有限，激进剪枝很重要
study = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(n_startup_trials=5),     # 少一点初始探索
    pruner=HyperbandPruner(                      # Hyperband 比 Median 更激进，适合 DL
        min_resource=3,                          # 至少跑 3 个 epoch 再判断
        reduction_factor=3
    )
)
study.optimize(objective, n_trials=30)  # DL 场景 30~50 个 trial 就差不多了
```

> **本章后续内容**：假设你已经判断 Optuna 对你的 DL 场景值得投入。下面教你怎么做。

---

## 5.1 PyTorch + Optuna 的基本模式

PyTorch 的灵活性意味着你需要**手动**在训练循环中插入 Optuna 的 API：

```python
for epoch in range(EPOCHS):
    train(...)           # 训练一个 epoch
    val_accuracy = ...   # 验证

    trial.report(val_accuracy, epoch)  # 报告中间结果

    if trial.should_prune():           # 检查是否剪枝
        raise optuna.TrialPruned()
```

---

## 5.2 完整训练流程

### Step 1：定义模型（搜索空间在模型结构中）

```python
class DynamicNet(nn.Module):
    def __init__(self, trial):
        super().__init__()
        # 搜索层数
        n_layers = trial.suggest_int('n_layers', 1, 3)
        in_features = 28 * 28
        self.layers = nn.ModuleList()

        for i in range(n_layers):
            out_features = trial.suggest_int(f'n_units_l{i}', 64, 256, step=64)
            self.layers.append(nn.Linear(in_features, out_features))
            self.layers.append(nn.ReLU())
            dropout = trial.suggest_float(f'dropout_l{i}', 0.0, 0.5)
            self.layers.append(nn.Dropout(dropout))
            in_features = out_features

        self.layers.append(nn.Linear(in_features, 10))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)
```

### Step 2：定义优化器（搜索空间在优化器配置中）

```python
def get_optimizer(trial, model):
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)

    if optimizer_name == 'Adam':
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        momentum = trial.suggest_float('momentum', 0.0, 0.99)
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:
        return optim.RMSprop(model.parameters(), lr=lr)
```

### Step 3：完整的 Objective 函数

```python
def objective(trial):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 模型
    model = DynamicNet(trial).to(device)
    optimizer = get_optimizer(trial, model)
    criterion = nn.CrossEntropyLoss()

    # 数据
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    train_loader, val_loader = get_data_loaders(batch_size)

    # 训练
    best_val_acc = 0.0
    for epoch in range(20):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                pred = model(data).argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        val_acc = correct / total
        best_val_acc = max(best_val_acc, val_acc)

        # 剪枝
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_acc
```

---

## 5.3 保存和加载最佳模型

Optuna 只保存参数，不保存模型权重。你需要手动保存：

### 方式一：在 Study 完成后保存

```python
study.optimize(objective, n_trials=100)

# 用最佳参数重新训练并保存
best_params = study.best_params
model = create_model(best_params)
# 完整训练...
torch.save(model.state_dict(), 'best_model.pth')
```

### 方式二：用 Callback 在优化过程中保存

```python
def save_best_callback(study, trial):
    if study.best_trial.number == trial.number:
        # 当前 trial 是最佳的，保存模型
        torch.save(trial.user_attrs['model_state'], 'best_model.pth')

# 在 objective 中把模型状态存入 trial
# trial.set_user_attr('model_state', model.state_dict())

study.optimize(objective, n_trials=100, callbacks=[save_best_callback])
```

### 方式三：用 Pickle 保存整个 Study

```python
import joblib
joblib.dump(study, 'study.pkl')

# 加载
study = joblib.load('study.pkl')
```

---

## 5.4 学习率调度器的搜索

```python
scheduler_name = trial.suggest_categorical('scheduler', ['none', 'step', 'cosine', 'plateau'])

if scheduler_name == 'step':
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
elif scheduler_name == 'cosine':
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
elif scheduler_name == 'plateau':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)

# 训练循环中
if scheduler_name != 'none':
    if scheduler_name == 'plateau':
        scheduler.step(val_acc)
    else:
        scheduler.step()
```

---

## 5.5 本章实战任务

1. 运行本章代码，在 MNIST 上搜索最优网络结构
2. 修改搜索空间：增加卷积层选项（CNN vs MLP）
3. 尝试搜索学习率调度器
4. 实现模型保存逻辑，确保能找到最优参数对应的权重

---

## 5.6 常见问题

**Q：GPU 内存不够，trial 之间不释放怎么办？**
A：在每个 trial 结束时手动清理：
```python
import gc
torch.cuda.empty_cache()
gc.collect()
```

**Q：多卡训练怎么集成 Optuna？**
A：Optuna 控制的是超参数搜索，和数据并行（DataParallel / DDP）是独立的。每个 trial 内部可以用 DDP，但不同 trial 之间不需要通信。

**Q：训练不稳定导致剪枝误杀好参数？**
A：增加 `n_warmup_steps`，或使用 `PatientPruner`。

---

## 5.7 本章小结

- PyTorch 中需要在 epoch 循环内手动调用 `report()` 和 `should_prune()`
- 搜索空间可以覆盖**模型结构**（层数、宽度）、**优化器配置**、**训练配置**（batch size、scheduler）
- Optuna 不自动保存模型权重，需要手动处理
- 清理 GPU 缓存是长时间搜索的必要步骤

**下一步**：学习条件搜索空间和多目标优化。
