"""
第6章代码：复杂搜索空间与多目标优化
运行方式：python code/06_complex_search_space.py

演示：
1. 条件参数（不同优化器有不同参数）
2. 多模型选择
3. 多目标优化（准确率 + 推理速度）
"""

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
# 示例1：条件参数（不同优化器有不同专属参数）
# ============================================
def demo_conditional_parameters():
    print("=" * 60)
    print("示例1：条件参数 - 不同优化器的专属参数")
    print("=" * 60)

    def objective(trial):
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'AdamW'])
        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)

        # 条件参数：只有选特定优化器时才出现
        if optimizer_name == 'Adam':
            beta1 = trial.suggest_float('beta1', 0.5, 0.99)
            beta2 = trial.suggest_float('beta2', 0.9, 0.999)
            eps = trial.suggest_float('eps', 1e-8, 1e-6, log=True)
            print(f"  [Trial {trial.number}] Adam -> lr={lr:.6f}, beta1={beta1:.4f}, beta2={beta2:.4f}")

        elif optimizer_name == 'SGD':
            momentum = trial.suggest_float('momentum', 0.0, 0.99)
            nesterov = trial.suggest_categorical('nesterov', [True, False])
            print(f"  [Trial {trial.number}] SGD -> lr={lr:.6f}, momentum={momentum:.4f}, nesterov={nesterov}")

        else:  # AdamW
            weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)
            print(f"  [Trial {trial.number}] AdamW -> lr={lr:.6f}, weight_decay={weight_decay:.6f}")

        # 模拟训练结果
        if optimizer_name == 'Adam':
            score = 0.85 + 0.1 * (lr / 0.1) + 0.05 * beta1
        elif optimizer_name == 'SGD':
            score = 0.80 + 0.12 * (lr / 0.1) + 0.08 * momentum
        else:
            score = 0.87 + 0.1 * (lr / 0.1) - 0.02 * weight_decay

        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=15)

    print(f"\n最佳结果: {study.best_value:.4f}")
    print(f"最佳参数: {study.best_params}")
    print()


# ============================================
# 示例2：多模型选择
# ============================================
def demo_model_selection():
    print("=" * 60)
    print("示例2：多模型选择")
    print("=" * 60)

    class SmallMLP(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 10)
            )
        def forward(self, x):
            return self.net(x)

    class DeepMLP(nn.Module):
        def __init__(self, h1, h2, dropout):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, h1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(h1, h2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(h2, 10)
            )
        def forward(self, x):
            return self.net(x)

    def objective(trial):
        model_type = trial.suggest_categorical('model_type', ['small', 'deep'])
        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
        batch_size = trial.suggest_categorical('batch_size', [64, 128])

        if model_type == 'small':
            hidden = trial.suggest_int('small_hidden', 64, 256, step=64)
            model = SmallMLP(hidden).to(device)
        else:
            h1 = trial.suggest_int('deep_h1', 64, 256, step=64)
            h2 = trial.suggest_int('deep_h2', 32, 128, step=32)
            dropout = trial.suggest_float('dropout', 0.0, 0.5)
            model = DeepMLP(h1, h2, dropout).to(device)

        train_loader, val_loader = get_data_loaders(batch_size)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # 快速训练3个epoch
        for epoch in range(3):
            model.train()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        # 评估
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                pred = model(data).argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        return correct / total

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20, show_progress_bar=True)

    print(f"\n最佳准确率: {study.best_value:.4f}")
    print(f"最佳参数: {study.best_params}")

    # 统计模型选择偏好
    model_counts = {}
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            m = t.params.get('model_type')
            model_counts[m] = model_counts.get(m, 0) + 1
    print(f"模型选择统计: {model_counts}")
    print()


# ============================================
# 示例3：多目标优化
# ============================================
def demo_multi_objective():
    print("=" * 60)
    print("示例3：多目标优化（最大化准确率，最小化推理时间）")
    print("=" * 60)

    def objective(trial):
        hidden = trial.suggest_int('hidden', 64, 512, step=64)
        num_layers = trial.suggest_int('num_layers', 1, 4)
        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)

        # 模拟准确率：越大越好
        # 层数适中、隐藏层适中时准确率最高
        accuracy = 0.7 + 0.15 * (1 - abs(hidden - 256) / 256)
        accuracy += 0.1 * (1 - abs(num_layers - 2) / 3)
        accuracy += 0.05 * (lr / 0.1)

        # 模拟推理时间：越小越好
        # 参数越多，推理时间越长
        params = hidden * hidden * num_layers
        inference_time = 1.0 + params / 50000

        return accuracy, inference_time

    study = optuna.create_study(
        directions=['maximize', 'minimize'],
        study_name='multi_objective_demo'
    )
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print(f"\nPareto 前沿解数量: {len(study.best_trials)}")
    print("前5个 Pareto 最优解:")
    for i, trial in enumerate(study.best_trials[:5]):
        acc, inf_time = trial.values
        print(f"  解 {i+1}: 准确率={acc:.4f}, 推理时间={inf_time:.4f}ms")
        print(f"    参数: hidden={trial.params['hidden']}, layers={trial.params['num_layers']}, lr={trial.params['lr']:.4f}")

    # 尝试可视化
    try:
        import matplotlib
        matplotlib.use('Agg')  # 无GUI环境
        import matplotlib.pyplot as plt

        all_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        accs = [t.values[0] for t in all_trials]
        times = [t.values[1] for t in all_trials]

        pareto_accs = [t.values[0] for t in study.best_trials]
        pareto_times = [t.values[1] for t in study.best_trials]

        plt.figure(figsize=(8, 6))
        plt.scatter(times, accs, alpha=0.5, label='All trials')
        plt.scatter(pareto_times, pareto_accs, color='red', s=100, label='Pareto front', zorder=5)
        plt.xlabel('Inference Time (ms)')
        plt.ylabel('Accuracy')
        plt.title('Multi-Objective Optimization: Pareto Front')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('pareto_front.png', dpi=150)
        print("\nPareto 前沿图已保存到 pareto_front.png")
    except Exception as e:
        print(f"\n可视化失败（可能缺少 matplotlib）: {e}")
    print()


def main():
    print("\n" + "=" * 60)
    print("复杂搜索空间与多目标优化")
    print("=" * 60)

    demo_conditional_parameters()
    demo_model_selection()
    demo_multi_objective()

    print("全部演示完成！")


if __name__ == '__main__':
    main()
