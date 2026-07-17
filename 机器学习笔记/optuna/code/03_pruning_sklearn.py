"""
第3章代码（sklearn版）：剪枝策略实战
运行方式：python code/03_pruning_sklearn.py

对比：
1. 无剪枝的基准
2. MedianPruner
3. HyperbandPruner
"""

import optuna
from optuna.pruners import MedianPruner, HyperbandPruner, PatientPruner
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
import time


X, y = load_digits(return_X_y=True)


def objective_no_prune(trial):
    """无剪枝版本（基准）"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 200),
        'max_depth': trial.suggest_int('max_depth', 2, 30),
        'min_samples_split': trial.suggest_float('min_samples_split', 0.01, 0.5),
    }

    clf = RandomForestClassifier(random_state=42, n_jobs=1, **params)
    scores = []
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(X, y):
        clf.fit(X[train_idx], y[train_idx])
        scores.append(clf.score(X[val_idx], y[val_idx]))

    return np.mean(scores)


def objective_with_prune(trial):
    """带剪枝版本 — 每完成一个 fold 就检查是否剪枝"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 200),
        'max_depth': trial.suggest_int('max_depth', 2, 30),
        'min_samples_split': trial.suggest_float('min_samples_split', 0.01, 0.5),
    }

    scores = []
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for step, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        clf = RandomForestClassifier(random_state=42, n_jobs=1, **params)
        clf.fit(X[train_idx], y[train_idx])
        score = clf.score(X[val_idx], y[val_idx])
        scores.append(score)

        # 报告中间结果
        trial.report(np.mean(scores), step)

        # 检查是否应该剪枝
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(scores)


def run_experiment(name, objective, pruner=None, n_trials=50):
    """运行实验并统计结果。"""
    print(f"\n{'='*60}")
    print(f"实验: {name}")
    print(f"{'='*60}")

    study = optuna.create_study(
        direction='maximize',
        pruner=pruner,
        study_name=name
    )

    start = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    elapsed = time.time() - start

    # 统计
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    print(f"总试验数: {len(study.trials)}")
    print(f"完成 (未剪枝): {len(completed)}")
    print(f"被剪枝: {len(pruned)} ({len(pruned)/len(study.trials)*100:.1f}%)")
    print(f"最佳准确率: {study.best_value:.6f}")
    print(f"最佳参数: {study.best_params}")
    print(f"总耗时: {elapsed:.1f} 秒")

    return {
        'name': name,
        'best_value': study.best_value,
        'completed': len(completed),
        'pruned': len(pruned),
        'elapsed': elapsed,
    }


def main():
    n_trials = 50

    # 1. 无剪枝基准
    baseline = run_experiment('无剪枝基准', objective_no_prune, n_trials=n_trials)

    # 2. MedianPruner
    median_result = run_experiment(
        'MedianPruner',
        objective_with_prune,
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        n_trials=n_trials
    )

    # 3. HyperbandPruner
    hyperband_result = run_experiment(
        'HyperbandPruner',
        objective_with_prune,
        pruner=HyperbandPruner(min_resource=1, reduction_factor=3),
        n_trials=n_trials
    )

    # 4. PatientPruner（包装 MedianPruner）
    patient_result = run_experiment(
        'PatientPruner(Median)',
        objective_with_prune,
        pruner=PatientPruner(MedianPruner(), patience=1),
        n_trials=n_trials
    )

    # 汇总
    print("\n" + "=" * 60)
    print("剪枝效果汇总")
    print("=" * 60)
    print(f"{'配置':<25} {'最佳值':>10} {'完成':>6} {'剪枝':>6} {'耗时(秒)':>10} {'加速比':>8}")
    print("-" * 60)

    for r in [baseline, median_result, hyperband_result, patient_result]:
        speedup = baseline['elapsed'] / r['elapsed'] if r['elapsed'] > 0 else 0
        print(f"{r['name']:<25} {r['best_value']:>10.6f} {r['completed']:>6} {r['pruned']:>6} {r['elapsed']:>10.1f} {speedup:>8.2f}x")

    print(f"\n结论：")
    print(f"- Hyperband 通常节省最多时间，但可能更激进地剪掉一些 trial")
    print(f"- Median 是最稳妥的选择，平衡了效率和效果")
    print(f"- PatientPruner 适合指标波动大的场景")


if __name__ == '__main__':
    main()
