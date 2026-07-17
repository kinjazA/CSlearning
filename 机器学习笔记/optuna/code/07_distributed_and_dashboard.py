"""
第7章代码：分布式优化与可视化
运行方式：python code/07_distributed_and_dashboard.py

演示：
1. SQLite storage + 多进程并行搜索
2. 超参数重要性分析
3. 结果导出
"""

import optuna
from optuna.samplers import TPESampler
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from multiprocessing import Process
import time
import os


X, y = load_digits(return_X_y=True)
DB_PATH = 'sqlite:///optuna_distributed.db'
STUDY_NAME = 'distributed_digits'


def objective(trial):
    """目标函数：随机森林超参搜索。"""
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 2, 30)
    min_samples_split = trial.suggest_float('min_samples_split', 0.01, 0.5)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        max_features=max_features,
        bootstrap=bootstrap,
        random_state=42,
        n_jobs=1  # 每个trial单线程，避免多进程争用
    )

    score = cross_val_score(clf, X, y, cv=3, scoring='accuracy').mean()
    return score


def worker(process_id, n_trials):
    """工作进程：每个进程独立加载 study 并优化。"""
    print(f"[Worker {process_id}] 启动，将运行 {n_trials} 个 trials")

    study = optuna.load_study(
        study_name=STUDY_NAME,
        storage=DB_PATH
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"[Worker {process_id}] 完成")


def demo_single_process():
    """单进程基准。"""
    print("=" * 60)
    print("单进程基准（100 trials）")
    print("=" * 60)

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42)
    )

    start = time.time()
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    elapsed = time.time() - start

    print(f"最佳准确率: {study.best_value:.6f}")
    print(f"总耗时: {elapsed:.1f} 秒")
    return elapsed


def demo_distributed():
    """多进程分布式搜索。"""
    print("\n" + "=" * 60)
    print("多进程分布式搜索（4进程 x 25 trials = 100 total）")
    print("=" * 60)

    # 删除旧数据库（如果存在）
    db_file = 'optuna_distributed.db'
    if os.path.exists(db_file):
        os.remove(db_file)

    # 创建 study（只需一次）
    optuna.create_study(
        study_name=STUDY_NAME,
        storage=DB_PATH,
        direction='maximize',
        sampler=TPESampler(seed=42),
        load_if_exists=True
    )

    start = time.time()

    # 启动4个进程
    processes = []
    for i in range(4):
        p = Process(target=worker, args=(i, 25))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    elapsed = time.time() - start

    # 加载结果
    study = optuna.load_study(study_name=STUDY_NAME, storage=DB_PATH)
    print(f"最佳准确率: {study.best_value:.6f}")
    print(f"总试验数: {len(study.trials)}")
    print(f"总耗时: {elapsed:.1f} 秒")

    return elapsed, study


def analyze_results(study):
    """分析搜索结果。"""
    print("\n" + "=" * 60)
    print("结果分析")
    print("=" * 60)

    # 1. 超参数重要性
    print("\n超参数重要性 (fANOVA):")
    try:
        importances = optuna.importance.get_param_importances(study)
        for param, importance in importances.items():
            bar = '█' * int(importance * 40)
            print(f"  {param:20s} {importance:.4f} {bar}")
    except Exception as e:
        print(f"  无法计算重要性: {e}")

    # 2. 前10个最佳 trial
    print("\n前10个最佳 trial:")
    sorted_trials = sorted(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: t.value,
        reverse=True
    )
    for i, t in enumerate(sorted_trials[:10]):
        print(f"  #{t.number:3d}: {t.value:.6f} | n_est={t.params.get('n_estimators'):3d}, depth={t.params.get('max_depth'):2d}, split={t.params.get('min_samples_split'):.3f}")

    # 3. 导出为 CSV
    print("\n导出结果到 CSV...")
    df = study.trials_dataframe()
    df.to_csv('distributed_results.csv', index=False)
    print(f"  已保存到 distributed_results.csv (行数: {len(df)})")

    # 4. 导出最佳参数
    import json
    with open('best_distributed_params.json', 'w') as f:
        json.dump(study.best_params, f, indent=2)
    print(f"  最佳参数已保存到 best_distributed_params.json")

    # 5. Dashboard 启动提示
    print("\n" + "=" * 60)
    print("可视化")
    print("=" * 60)
    print("启动 Dashboard 查看可视化结果：")
    print(f"  optuna-dashboard {DB_PATH}")
    print("然后在浏览器打开 http://localhost:8080")
    print("\nDashboard 中可以查看：")
    print("  - 优化历史曲线")
    print("  - 超参数重要性图")
    print("  - 平行坐标图")
    print("  - 轮廓图（参数间交互作用）")


def main():
    print("\n" + "=" * 60)
    print("分布式优化与可视化")
    print("=" * 60)

    # 单进程基准
    t1 = demo_single_process()

    # 多进程分布式
    t2, study = demo_distributed()

    # 对比
    print("\n" + "=" * 60)
    print("性能对比")
    print("=" * 60)
    print(f"单进程: {t1:.1f} 秒")
    print(f"4进程:  {t2:.1f} 秒")
    if t2 > 0:
        print(f"加速比: {t1/t2:.2f}x")

    # 分析结果
    analyze_results(study)

    print("\n演示完成！")


if __name__ == '__main__':
    main()
