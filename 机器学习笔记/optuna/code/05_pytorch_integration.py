"""
第5章代码：PyTorch 集成实战
运行方式：python code/05_pytorch_integration.py

在 MNIST 上训练神经网络，搜索网络结构、优化器配置和训练参数。
"""

import optuna
from optuna.pruners import HyperbandPruner
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import time
import gc


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# ============================================
# 数据准备
# ============================================
def get_data_loaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# ============================================
# 动态网络定义（搜索空间在结构中）
# ============================================
class DynamicNet(nn.Module):
    def __init__(self, trial):
        super().__init__()
        n_layers = trial.suggest_int('n_layers', 1, 3)
        in_features = 28 * 28
        layers = []

        for i in range(n_layers):
            out_features = trial.suggest_int(f'n_units_l{i}', 64, 256, step=64)
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            dropout_rate = trial.suggest_float(f'dropout_l{i}', 0.0, 0.5)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_features = out_features

        layers.append(nn.Linear(in_features, 10))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)


def create_optimizer(trial, model):
    """搜索优化器配置。"""
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


def get_scheduler(trial, optimizer):
    """搜索学习率调度器。"""
    scheduler_name = trial.suggest_categorical('scheduler', ['none', 'step', 'cosine'])

    if scheduler_name == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    elif scheduler_name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    return None


# ============================================
# 训练与验证
# ============================================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return correct / total


# ============================================
# Objective 函数
# ============================================
def objective(trial):
    # 超参数
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])

    # 模型
    model = DynamicNet(trial).to(device)
    optimizer = create_optimizer(trial, model)
    scheduler = get_scheduler(trial, optimizer)
    criterion = nn.CrossEntropyLoss()

    train_loader, val_loader = get_data_loaders(batch_size)

    best_val_acc = 0.0

    for epoch in range(15):  # 最多15个epoch
        train_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model, val_loader, device)
        best_val_acc = max(best_val_acc, val_acc)

        if scheduler:
            scheduler.step()

        # 报告中间结果
        trial.report(val_acc, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

    # 清理 GPU 内存
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return best_val_acc


# ============================================
# 保存最佳模型
# ============================================
def save_best_model(study, trial):
    """Callback：当找到新的最佳 trial 时保存模型。"""
    if study.best_trial.number == trial.number:
        # 注意：这里我们只是记录最佳参数，不保存权重
        # 实际项目中可以用 trial.user_attrs 传递模型状态
        print(f"  >>> 新的最佳结果！Trial #{trial.number}: {trial.value:.4f}")


def main():
    print("\n" + "=" * 60)
    print("PyTorch + Optuna 集成实战")
    print("=" * 60)

    study = optuna.create_study(
        direction='maximize',
        pruner=HyperbandPruner(min_resource=1, reduction_factor=3),
        study_name='pytorch_mnist'
    )

    start = time.time()
    study.optimize(objective, n_trials=50, show_progress_bar=True, callbacks=[save_best_model])
    elapsed = time.time() - start

    # 结果统计
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    print("\n" + "=" * 60)
    print("搜索结果")
    print("=" * 60)
    print(f"总试验数: 50")
    print(f"完成: {len(completed)}")
    print(f"被剪枝: {len(pruned)} ({len(pruned)/50*100:.1f}%)")
    print(f"最佳验证准确率: {study.best_value:.4f}")
    print(f"最佳参数:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"总耗时: {elapsed:.1f} 秒")

    # 超参数重要性分析
    print("\n" + "=" * 60)
    print("超参数重要性 (fANOVA)")
    print("=" * 60)
    try:
        importances = optuna.importance.get_param_importances(study)
        for param, importance in importances.items():
            print(f"  {param}: {importance:.4f}")
    except Exception as e:
        print(f"  重要性分析需要更多 completed trial，当前完成数: {len(completed)}")
        print(f"  错误: {e}")

    # 保存最佳参数
    import json
    with open('best_pytorch_params.json', 'w') as f:
        json.dump(study.best_params, f, indent=2)
    print("\n最佳参数已保存到 best_pytorch_params.json")


if __name__ == '__main__':
    main()
