"""
第8章代码（sklearn版）：综合实战项目
运行方式：python code/08_project_sklearn.py

项目：California Housing 房价预测回归
- 搜索空间：5+ 参数
- Sampler: TPE + multivariate
- Pruner: Hyperband
- Trials: 100
"""

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import numpy as np
import time
import json


# 加载数据
print("加载 California Housing 数据集...")
housing = fetch_california_housing()
X, y = housing.data, housing.target
print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}")


def objective(trial):
    """完整的目标函数，包含条件参数和多模型选择。"""

    # ========== 预处理 ==========
    scaler_type = trial.suggest_categorical('scaler', ['standard', 'minmax', 'none'])
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = 'passthrough'

    # ========== 模型选择 ==========
    model_type = trial.suggest_categorical('model', ['rf', 'gb'])

    if model_type == 'rf':
        model = RandomForestRegressor(
            n_estimators=trial.suggest_int('rf_n_estimators', 50, 300),
            max_depth=trial.suggest_int('rf_max_depth', 3, 20),
            min_samples_split=trial.suggest_float('rf_min_samples_split', 0.01, 0.5),
            min_samples_leaf=trial.suggest_int('rf_min_samples_leaf', 1, 10),
            max_features=trial.suggest_categorical('rf_max_features', ['sqrt', 'log2', None]),
            random_state=42,
            n_jobs=-1
        )
    else:  # GradientBoosting
        model = GradientBoostingRegressor(
            n_estimators=trial.suggest_int('gb_n_estimators', 50, 300),
            max_depth=trial.suggest_int('gb_max_depth', 2, 10),
            learning_rate=trial.suggest_float('gb_lr', 1e-3, 0.3, log=True),
            subsample=trial.suggest_float('gb_subsample', 0.5, 1.0),
            random_state=42
        )

    # ========== Pipeline + CV with Pruning ==========
    pipeline = Pipeline([
        ('scaler', scaler),
        ('model', model)
    ])

    # 手动 CV 以支持剪枝
    scores = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for step, (train_idx, val_idx) in enumerate(kf.split(X)):
        pipeline.fit(X[train_idx], y[train_idx])
        y_pred = pipeline.predict(X[val_idx])
        rmse = np.sqrt(mean_squared_error(y[val_idx], y_pred))
        scores.append(rmse)

        # 报告中间结果（RMSE 越小越好）
        trial.report(np.mean(scores), step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(scores)


def main():
    print("\n" + "=" * 70)
    print("综合实战项目：California Housing 超参数优化")
    print("=" * 70)
    print("配置：")
    print("  - Sampler: TPESampler(multivariate=True)")
    print("  - Pruner: HyperbandPruner")
    print("  - Trials: 100")
    print("  - 搜索空间: 5+ 参数，包含条件参数")
    print()

    # 创建 study
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(multivariate=True, seed=42),
        pruner=HyperbandPruner(min_resource=1, reduction_factor=3),
        study_name='california_housing_project'
    )

    # 运行优化
    start = time.time()
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    elapsed = time.time() - start

    # ========== 结果统计 ==========
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    print("\n" + "=" * 70)
    print("实验结果")
    print("=" * 70)
    print(f"总试验数: 100")
    print(f"完成: {len(completed)}")
    print(f"被剪枝: {len(pruned)} ({len(pruned)/100*100:.1f}%)")
    print(f"总耗时: {elapsed:.1f} 秒")
    print(f"\n最佳 RMSE: {study.best_value:.6f}")
    print(f"最佳参数:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # ========== 超参数重要性 ==========
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

    # ========== 收敛分析 ==========
    print("\n" + "=" * 70)
    print("收敛分析")
    print("=" * 70)
    values = [t.value for t in completed if t.value is not None]
    cumulative_best = []
    current_best = float('inf')
    for v in values:
        current_best = min(current_best, v)
        cumulative_best.append(current_best)

    checkpoints = [10, 25, 50, 75, 100]
    for cp in checkpoints:
        if cp <= len(cumulative_best):
            print(f"  第{cp:3d}次 trial 时的最佳 RMSE: {cumulative_best[cp-1]:.6f}")

    # ========== 保存结果 ==========
    print("\n" + "=" * 70)
    print("保存结果")
    print("=" * 70)

    # 最佳参数
    with open('project_sklearn_best_params.json', 'w') as f:
        json.dump(study.best_params, f, indent=2)
    print("  最佳参数: project_sklearn_best_params.json")

    # CSV
    df = study.trials_dataframe()
    df.to_csv('project_sklearn_results.csv', index=False)
    print(f"  完整结果: project_sklearn_results.csv ({len(df)} 行)")

    # 实验报告
    report = f"""# 实验报告：California Housing 超参数优化

## 实验设置
- 数据集: California Housing (n={X.shape[0]}, features={X.shape[1]})
- Sampler: TPESampler(multivariate=True)
- Pruner: HyperbandPruner
- Trials: 100
- 搜索空间:
  - scaler: ['standard', 'minmax', 'none']
  - model: ['rf', 'gb']
  - 条件参数: 根据 model 类型有不同的参数

## 最佳结果
- 最佳 RMSE: {study.best_value:.6f}
- 最佳参数: {json.dumps(study.best_params, indent=2)}

## 统计
- 完成 trials: {len(completed)}
- 剪枝 trials: {len(pruned)} ({len(pruned)/100*100:.1f}%)
- 总耗时: {elapsed:.1f} 秒

## 收敛情况
- 第10次 trial: {cumulative_best[9]:.6f}
- 第50次 trial: {cumulative_best[49]:.6f}
- 第100次 trial: {cumulative_best[99]:.6f}
"""
    with open('project_sklearn_report.md', 'w') as f:
        f.write(report)
    print("  实验报告: project_sklearn_report.md")

    print("\n" + "=" * 70)
    print("项目完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
