"""
第4章代码：scikit-learn 集成实战
运行方式：python code/04_sklearn_integration.py

演示三种集成方式：
1. OptunaSearchCV（最简单）
2. 原生 optimize（推荐）
3. 带剪枝的手动 CV（最高效）
"""

import optuna
from optuna.integration import OptunaSearchCV
from optuna.pruners import MedianPruner
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
import numpy as np
import time


X, y = load_wine(return_X_y=True)


# ============================================
# 方式一：OptunaSearchCV（最简单，不支持剪枝）
# ============================================
def demo_optuna_search_cv():
    print("=" * 60)
    print("方式一：OptunaSearchCV")
    print("=" * 60)

    model = RandomForestClassifier(random_state=42)

    param_distributions = {
        'n_estimators': optuna.distributions.IntDistribution(50, 300),
        'max_depth': optuna.distributions.IntDistribution(3, 20),
        'min_samples_split': optuna.distributions.FloatDistribution(0.01, 0.5),
    }

    optuna_search = OptunaSearchCV(
        model,
        param_distributions,
        n_trials=50,
        cv=5,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    start = time.time()
    optuna_search.fit(X, y)
    elapsed = time.time() - start

    print(f"最佳准确率: {optuna_search.best_score_:.6f}")
    print(f"最佳参数: {optuna_search.best_params_}")
    print(f"总耗时: {elapsed:.1f} 秒")
    print()
    return elapsed


# ============================================
# 方式二：原生 optimize（最灵活，推荐日常使用）
# ============================================
def demo_native_optimize():
    print("=" * 60)
    print("方式二：原生 optimize")
    print("=" * 60)

    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_samples_split = trial.suggest_float('min_samples_split', 0.01, 0.5)

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1
        )

        score = cross_val_score(clf, X, y, cv=5, scoring='accuracy', n_jobs=-1).mean()
        return score

    study = optuna.create_study(direction='maximize')

    start = time.time()
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    elapsed = time.time() - start

    print(f"最佳准确率: {study.best_value:.6f}")
    print(f"最佳参数: {study.best_params}")
    print(f"总耗时: {elapsed:.1f} 秒")
    print()
    return elapsed


# ============================================
# 方式三：带剪枝的手动 CV（最高效）
# ============================================
def demo_pruned_cv():
    print("=" * 60)
    print("方式三：带剪枝的手动 CV")
    print("=" * 60)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_float('min_samples_split', 0.01, 0.5),
        }

        scores = []
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for step, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            clf = RandomForestClassifier(random_state=42, n_jobs=1, **params)
            clf.fit(X[train_idx], y[train_idx])
            score = clf.score(X[val_idx], y[val_idx])
            scores.append(score)

            trial.report(np.mean(scores), step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(scores)

    study = optuna.create_study(
        direction='maximize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    )

    start = time.time()
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    elapsed = time.time() - start

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    print(f"最佳准确率: {study.best_value:.6f}")
    print(f"最佳参数: {study.best_params}")
    print(f"完成: {len(completed)}, 剪枝: {len(pruned)}")
    print(f"总耗时: {elapsed:.1f} 秒")
    print()
    return elapsed


# ============================================
# 进阶：多模型选择的 Pipeline
# ============================================
def demo_multi_model_pipeline():
    print("=" * 60)
    print("进阶：多模型选择的 Pipeline")
    print("=" * 60)

    def objective(trial):
        # 预处理选择
        scaler_type = trial.suggest_categorical('scaler', ['standard', 'minmax', 'none'])
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = 'passthrough'

        # 模型选择
        model_type = trial.suggest_categorical('model', ['rf', 'gb', 'svm'])

        if model_type == 'rf':
            model = RandomForestClassifier(
                n_estimators=trial.suggest_int('rf_n_estimators', 50, 300),
                max_depth=trial.suggest_int('rf_max_depth', 3, 20),
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gb':
            model = GradientBoostingClassifier(
                n_estimators=trial.suggest_int('gb_n_estimators', 50, 300),
                learning_rate=trial.suggest_float('gb_lr', 0.01, 0.3, log=True),
                max_depth=trial.suggest_int('gb_max_depth', 2, 10),
                random_state=42
            )
        else:  # svm
            model = SVC(
                C=trial.suggest_float('svm_C', 1e-3, 1e3, log=True),
                kernel=trial.suggest_categorical('svm_kernel', ['rbf', 'linear']),
                gamma='scale',
                random_state=42
            )

        # 构建 Pipeline
        pipeline = Pipeline([
            ('scaler', scaler),
            ('model', model)
        ])

        score = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy', n_jobs=-1).mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    print(f"最佳准确率: {study.best_value:.6f}")
    print(f"最佳参数:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # 统计每个模型被选中的次数
    model_counts = {}
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            m = t.params.get('model')
            model_counts[m] = model_counts.get(m, 0) + 1
    print(f"\n各模型被选中次数: {model_counts}")
    print()


def main():
    print("\n" + "=" * 60)
    print("scikit-learn 集成实战演示")
    print("=" * 60)
    print(f"数据集: Wine (样本数={X.shape[0]}, 特征数={X.shape[1]})")
    print()

    # 三种方式对比
    t1 = demo_optuna_search_cv()
    t2 = demo_native_optimize()
    t3 = demo_pruned_cv()

    print("=" * 60)
    print("三种方式耗时对比")
    print("=" * 60)
    print(f"OptunaSearchCV:     {t1:.1f} 秒")
    print(f"原生 optimize:      {t2:.1f} 秒")
    print(f"带剪枝的手动 CV:     {t3:.1f} 秒")

    # 多模型选择
    demo_multi_model_pipeline()

    print("\n全部演示完成！")


if __name__ == '__main__':
    main()
