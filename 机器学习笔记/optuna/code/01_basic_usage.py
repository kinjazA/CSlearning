"""
第1章代码：核心概念与基础用法
运行方式：python code/01_basic_usage.py
"""

import optuna


# ============================================
# 示例1：最简单的目标函数优化
# 优化 f(x) = (x - 2)^2，最小值在 x=2 时取得 0
# ============================================
def objective_simple(trial):
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2


print("=" * 50)
print("示例1：简单函数优化")
print("=" * 50)

study1 = optuna.create_study(direction='minimize')
study1.optimize(objective_simple, n_trials=100)

print(f"最佳值 (最小loss): {study1.best_value:.6f}")
print(f"最佳参数: {study1.best_params}")
print(f"最佳试验编号: {study1.best_trial.number}")
print(f"总试验数: {len(study1.trials)}")


# ============================================
# 示例2：多种参数类型的演示
# 模拟一个机器学习场景：学习率、批量大小、优化器、层数
# ============================================
def objective_ml_mock(trial):
    # 连续值（对数均匀分布，适合学习率）
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)

    # 整数
    batch_size = trial.suggest_int('batch_size', 16, 256, step=16)
    num_layers = trial.suggest_int('num_layers', 1, 5)

    # 类别
    optimizer = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])

    # 模拟的损失函数（参数越合理，损失越低）
    # 这里人为构造：学习率接近 0.01 最好，层数2-3最好
    loss = (
        abs(learning_rate - 0.01) * 100 +           # 学习率偏离 0.01 惩罚
        abs(num_layers - 2.5) * 2 +                  # 层数偏离 2-3 惩罚
        (0 if optimizer == 'Adam' else 1) +          # Adam 稍微好一点
        abs(batch_size - 64) * 0.01                  # batch_size 偏离 64 轻微惩罚
    )

    return loss


print("\n" + "=" * 50)
print("示例2：模拟ML超参数优化")
print("=" * 50)

study2 = optuna.create_study(direction='minimize')
study2.optimize(objective_ml_mock, n_trials=50)

print(f"最佳值: {study2.best_value:.6f}")
print(f"最佳参数:")
for key, value in study2.best_params.items():
    print(f"  {key}: {value}")


# ============================================
# 示例3：带约束的整数搜索（step参数）
# ============================================
def objective_step(trial):
    # 必须是 8 的倍数（适合 GPU 显存对齐）
    hidden_size = trial.suggest_int('hidden_size', 64, 512, step=8)
    # 必须是 2 的幂次（适合计算机架构）
    num_heads = trial.suggest_int('num_heads', 1, 16, log=True)

    # 模拟：hidden_size=256, num_heads=8 最佳
    score = abs(hidden_size - 256) + abs(num_heads - 8) * 10
    return score


print("\n" + "=" * 50)
print("示例3：步长约束的整数搜索")
print("=" * 50)

study3 = optuna.create_study(direction='minimize')
study3.optimize(objective_step, n_trials=30)

print(f"最佳参数: {study3.best_params}")
print(f"hidden_size 是 8 的倍数吗？{study3.best_params['hidden_size'] % 8 == 0}")


# ============================================
# 示例4：多目标优化入门（简单演示）
# 同时优化准确率和速度
# ============================================
def objective_multi(trial):
    learning_rate = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    model_size = trial.suggest_int('model_size', 64, 512, step=64)

    # 模拟：学习率越高、模型越大，准确率越高但速度越慢
    accuracy = 0.7 + 0.2 * (learning_rate / 0.1) + 0.1 * (model_size / 512)
    inference_time = 1.0 + 5.0 * (model_size / 512)

    return accuracy, inference_time  # 最大化准确率，最小化推理时间


print("\n" + "=" * 50)
print("示例4：多目标优化（最大化准确率，最小化推理时间）")
print("=" * 50)

study4 = optuna.create_study(
    directions=['maximize', 'minimize'],
    study_name='multi_objective_demo'
)
study4.optimize(objective_multi, n_trials=50)

# 获取 Pareto 前沿（非支配解集）
print(f"Pareto 前沿解数量: {len(study4.best_trials)}")
for i, trial in enumerate(study4.best_trials[:3]):
    print(f"  解 {i+1}: 准确率={trial.values[0]:.4f}, 推理时间={trial.values[1]:.4f}")
    print(f"    参数: {trial.params}")


# ============================================
# 示例5：Study 信息查看
# ============================================
print("\n" + "=" * 50)
print("示例5：Study 信息查看")
print("=" * 50)

# 获取所有 trial 的结果
print(f"Study 1 所有试验数: {len(study1.trials)}")
print(f"Study 1 成功完成的试验数: {len([t for t in study1.trials if t.state == optuna.trial.TrialState.COMPLETE])}")

# 查看最差的结果
worst_trial = max(study1.trials, key=lambda t: t.value if t.value is not None else float('-inf'))
print(f"最差试验 (#{worst_trial.number}): value={worst_trial.value:.4f}, params={worst_trial.params}")
