"""
第2章代码：采样器原理深入 —— 对比实验
运行方式：python code/02_sampler_comparison.py

本代码对比三种采样器在相同搜索空间和评估函数上的表现：
- TPESampler（贝叶斯优化）
- RandomSampler（随机搜索）
- CmaEsSampler（协方差矩阵适应进化策略）
"""

import optuna
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import time


# 加载数据集（用 digits 比 iris 更有区分度）
X, y = load_digits(return_X_y=True)


def create_objective():
    """创建一个闭包，返回 objective 函数。每个采样器需要独立的 objective。"""
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 10, 200)
        max_depth = trial.suggest_int('max_depth', 2, 30)
        min_samples_split = trial.suggest_float('min_samples_split', 0.01, 0.5)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
            random_state=42,
            n_jobs=-1
        )

        score = cross_val_score(clf, X, y, cv=3, scoring='accuracy', n_jobs=-1).mean()
        return score
    return objective


def run_study(sampler, sampler_name, n_trials):
    """运行一个 study 并返回结果统计。"""
    print(f"\n{'='*60}")
    print(f"采样器: {sampler_name} | Trials: {n_trials}")
    print(f"{'='*60}")

    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        study_name=f'{sampler_name}_{n_trials}'
    )

    start_time = time.time()
    study.optimize(create_objective(), n_trials=n_trials, show_progress_bar=True)
    elapsed = time.time() - start_time

    # 统计信息
    best_value = study.best_value
    best_params = study.best_params
    best_trial_num = study.best_trial.number

    # 计算前 N 次试验的最佳值变化（收敛曲线）
    trials = study.trials
    values = [t.value for t in trials if t.value is not None]
    cumulative_best = []
    current_best = float('-inf')
    for v in values:
        current_best = max(current_best, v)
        cumulative_best.append(current_best)

    print(f"最佳准确率: {best_value:.6f}")
    print(f"最佳参数: {best_params}")
    print(f"最佳试验编号: {best_trial_num}")
    print(f"总耗时: {elapsed:.1f} 秒")
    print(f"第10次试验时的最佳值: {cumulative_best[9]:.6f}")
    print(f"第30次试验时的最佳值: {cumulative_best[29]:.6f}")
    print(f"第50次试验时的最佳值: {cumulative_best[49]:.6f}")

    return {
        'sampler': sampler_name,
        'n_trials': n_trials,
        'best_value': best_value,
        'best_trial_num': best_trial_num,
        'elapsed': elapsed,
        'cumulative_best': cumulative_best,
    }


def main():
    results = []
    n_trials = 100

    # 1. TPE 采样器（默认）
    results.append(run_study(
        TPESampler(seed=42),
        'TPE',
        n_trials
    ))

    # 2. 随机采样器（基线）
    results.append(run_study(
        RandomSampler(seed=42),
        'Random',
        n_trials
    ))

    # 3. CMA-ES 采样器
    # 注意：CMA-ES 对整数/类别参数支持有限，这里主要测试连续参数部分
    results.append(run_study(
        CmaEsSampler(seed=42),
        'CMA-ES',
        n_trials
    ))

    # 4. TPE + multivariate（考虑参数相关性）
    results.append(run_study(
        TPESampler(seed=42, multivariate=True),
        'TPE_Multivariate',
        n_trials
    ))

    # 汇总对比
    print("\n" + "=" * 60)
    print("汇总对比")
    print("=" * 60)
    print(f"{'采样器':<20} {'最佳值':>10} {'最佳Trial':>10} {'耗时(秒)':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['sampler']:<20} {r['best_value']:>10.6f} {r['best_trial_num']:>10} {r['elapsed']:>10.1f}")

    # 收敛速度对比：看多少次 trial 达到各自最终 90% 的水平
    print("\n" + "=" * 60)
    print("收敛速度分析（达到最终最佳值 90% 所需的 trial 数）")
    print("=" * 60)
    for r in results:
        target = r['best_value'] - (r['best_value'] - 0.5) * 0.1  # 达到最终值90%水平的阈值
        for i, v in enumerate(r['cumulative_best']):
            if v >= target:
                print(f"{r['sampler']:<20}: 第 {i+1:>3} 次 trial")
                break


if __name__ == '__main__':
    main()
