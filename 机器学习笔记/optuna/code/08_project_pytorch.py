"""
第8章代码（PyTorch版）：综合实战项目
运行方式：python code/08_project_pytorch.py

项目：MNIST 图像分类
- 搜索空间：7+ 参数
- Sampler: TPE + multivariate
- Pruner: Hyperband
- Trials: 100
- 包含模型保存
"""

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import time
import json
import gc


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# ============================================
# 数据加载
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

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    )


# ============================================
# 模型定义（支持两种架构）
# ============================================
class MLPModel(nn.Module):
    """全连接网络。"""
    def __init__(self, hidden_sizes, dropout_rates):
        super().__init__()
        layers = []
        in_size = 28 * 28
        for hidden, drop in zip(hidden_sizes, dropout_rates):
            layers.append(nn.Linear(in_size, hidden))
            layers.append(nn.ReLU())
            if drop > 0:
                layers.append(nn.Dropout(drop))
            in_size = hidden
        layers.append(nn.Linear(in_size, 10))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x.view(x.size(0), -1))


class ConvModel(nn.Module):
    """简单的 CNN。"""
    def __init__(self, conv_channels, fc_size, dropout):
        super().__init__()
        layers = []
        in_ch = 1
        for out_ch in conv_channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ])
            in_ch = out_ch

        self.conv = nn.Sequential(*layers)
        # 经过若干次池化后计算特征图大小
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch * 7 * 7, fc_size),  # MNIST 28x28, 两次池化后 7x7
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_size, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)


def create_model(trial):
    """根据 trial 创建模型。"""
    model_type = trial.suggest_categorical('model_type', ['mlp', 'cnn'])

    if model_type == 'mlp':
        n_layers = trial.suggest_int('mlp_n_layers', 1, 3)
        hidden_sizes = []
        dropout_rates = []
        for i in range(n_layers):
            hidden_sizes.append(trial.suggest_int(f'mlp_hidden_{i}', 64, 512, step=64))
            dropout_rates.append(trial.suggest_float(f'mlp_dropout_{i}', 0.0, 0.5))
        return MLPModel(hidden_sizes, dropout_rates)
    else:
        n_conv = trial.suggest_int('cnn_n_conv', 1, 3)
        channels = []
        for i in range(n_conv):
            channels.append(trial.suggest_int(f'cnn_ch_{i}', 16, 64, step=16))
        fc_size = trial.suggest_int('cnn_fc_size', 64, 256, step=64)
        dropout = trial.suggest_float('cnn_dropout', 0.0, 0.5)
        return ConvModel(channels, fc_size, dropout)


def create_optimizer(trial, model):
    """创建优化器（条件参数）。"""
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)

    if optimizer_name == 'Adam':
        weight_decay = trial.suggest_float('adam_weight_decay', 1e-5, 1e-2, log=True)
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        momentum = trial.suggest_float('sgd_momentum', 0.0, 0.99)
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum)


# ============================================
# 训练和验证
# ============================================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
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
    correct = total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return correct / total


# ============================================
# Objective
# ============================================
best_model_state = None
best_model_params = None


def objective(trial):
    global best_model_state, best_model_params

    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])

    model = create_model(trial).to(device)
    optimizer = create_optimizer(trial, model)
    criterion = nn.CrossEntropyLoss()

    train_loader, val_loader = get_data_loaders(batch_size)

    best_val_acc = 0.0

    for epoch in range(20):
        train_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model, val_loader, device)
        best_val_acc = max(best_val_acc, val_acc)

        trial.report(val_acc, epoch)
        if trial.should_prune():
            del model, optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            raise optuna.TrialPruned()

    # 保存模型状态到 trial（用于后续保存最佳模型）
    trial.set_user_attr('model_state', {k: v.cpu().clone() for k, v in model.state_dict().items()})

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return best_val_acc


def save_callback(study, trial):
    """保存最佳模型的 callback。"""
    if study.best_trial.number == trial.number:
        state = trial.user_attrs.get('model_state')
        if state:
            torch.save(state, 'project_pytorch_best_model.pth')
            print(f"  >>> 保存最佳模型 (Trial #{trial.number}): {trial.value:.4f}")


# ============================================
# 主程序
# ============================================
def main():
    print("\n" + "=" * 70)
    print("综合实战项目：MNIST 神经网络超参数优化")
    print("=" * 70)
    print("配置：")
    print("  - 架构选择: MLP 或 CNN")
    print("  - Sampler: TPESampler(multivariate=True)")
    print("  - Pruner: HyperbandPruner")
    print("  - Trials: 100")
    print("  - 搜索空间: 7+ 参数，包含条件参数")
    print()

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(multivariate=True, seed=42),
        pruner=HyperbandPruner(min_resource=1, reduction_factor=3),
        study_name='mnist_pytorch_project'
    )

    start = time.time()
    study.optimize(objective, n_trials=100, show_progress_bar=True, callbacks=[save_callback])
    elapsed = time.time() - start

    # 结果统计
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    print("\n" + "=" * 70)
    print("实验结果")
    print("=" * 70)
    print(f"总试验数: 100")
    print(f"完成: {len(completed)}")
    print(f"被剪枝: {len(pruned)} ({len(pruned)/100*100:.1f}%)")
    print(f"总耗时: {elapsed:.1f} 秒")
    print(f"\n最佳验证准确率: {study.best_value:.4f}")
    print(f"最佳参数:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # 模型类型统计
    model_counts = {}
    for t in completed:
        m = t.params.get('model_type')
        model_counts[m] = model_counts.get(m, 0) + 1
    print(f"\n模型选择统计: {model_counts}")

    # 超参数重要性
    print("\n" + "=" * 70)
    print("超参数重要性")
    print("=" * 70)
    try:
        importances = optuna.importance.get_param_importances(study)
        for param, importance in importances.items():
            bar = '█' * int(importance * 40)
            print(f"  {param:25s} {importance:.4f} {bar}")
    except Exception as e:
        print(f"  计算失败: {e}")

    # 收敛分析
    print("\n" + "=" * 70)
    print("收敛分析")
    print("=" * 70)
    values = [t.value for t in completed if t.value is not None]
    cumulative_best = []
    current_best = 0.0
    for v in values:
        current_best = max(current_best, v)
        cumulative_best.append(current_best)

    for cp in [10, 25, 50, 75, 100]:
        if cp <= len(cumulative_best):
            print(f"  第{cp:3d}次 trial 时的最佳准确率: {cumulative_best[cp-1]:.4f}")

    # 保存结果
    print("\n" + "=" * 70)
    print("保存结果")
    print("=" * 70)

    with open('project_pytorch_best_params.json', 'w') as f:
        json.dump(study.best_params, f, indent=2)
    print("  最佳参数: project_pytorch_best_params.json")

    if best_model_state:
        print("  最佳模型: project_pytorch_best_model.pth")

    df = study.trials_dataframe()
    df.to_csv('project_pytorch_results.csv', index=False)
    print(f"  完整结果: project_pytorch_results.csv ({len(df)} 行)")

    # 实验报告
    report = f"""# 实验报告：MNIST 神经网络超参数优化

## 实验设置
- 数据集: MNIST
- Sampler: TPESampler(multivariate=True)
- Pruner: HyperbandPruner
- Trials: 100
- 架构选择: MLP / CNN（条件参数）

## 最佳结果
- 最佳验证准确率: {study.best_value:.4f}
- 最佳参数: {json.dumps(study.best_params, indent=2)}

## 统计
- 完成 trials: {len(completed)}
- 剪枝 trials: {len(pruned)} ({len(pruned)/100*100:.1f}%)
- 总耗时: {elapsed:.1f} 秒
- 模型选择: {json.dumps(model_counts)}

## 收敛情况
- 第10次 trial: {cumulative_best[9]:.4f}
- 第50次 trial: {cumulative_best[49]:.4f}
- 第100次 trial: {cumulative_best[99]:.4f}
"""
    with open('project_pytorch_report.md', 'w') as f:
        f.write(report)
    print("  实验报告: project_pytorch_report.md")

    print("\n" + "=" * 70)
    print("项目完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
