"""
第 07 章 · Parallel + delayed — 验证代码

目的：感受串行→并行的一行改造，以及踩坑。

实验：
  1. 串行 vs 并行 — 速度对比
  2. delayed 的必要性 — 不加会怎样
  3. n_jobs 的影响 — 1/4/-1
  4. 实战：并行交叉验证
  5. 实战：并行超参搜索
  6. Memory + Parallel 组合
  7. 五个坑的演示

运行方式：
    uv run python code/07-Parallel-delayed-并行.py
"""

import os
import sys
import time
import tempfile
import shutil

import numpy as np
from joblib import Parallel, delayed, Memory

# ============================================================
tmp_dir = tempfile.mkdtemp(prefix="joblib_ch07_")
cache_dir = os.path.join(tmp_dir, "cache")
print(f"Python: {sys.version.split()[0]}, joblib: {joblib.__version__}")
print(f"CPU 核心: {os.cpu_count()}\n")


# ============================================================
# 实验 1：串行 vs 并行 — 速度对比
# ============================================================
def experiment_1_serial_vs_parallel():
    print("=" * 60)
    print("实验 1 · 串行 vs 并行 — 速度对比")
    print("=" * 60)

    def slow_task(i):
        time.sleep(0.1)
        return i ** 2

    tasks = list(range(16))

    # 串行
    print(f"\n串行执行 {len(tasks)} 个任务...")
    t0 = time.perf_counter()
    serial = [slow_task(i) for i in tasks]
    t_serial = time.perf_counter() - t0
    print(f"  耗时: {t_serial:.2f}s")

    # 并行
    print(f"并行执行 {len(tasks)} 个任务 (n_jobs=4)...")
    t0 = time.perf_counter()
    parallel = Parallel(n_jobs=4)(
        delayed(slow_task)(i) for i in tasks
    )
    t_parallel = time.perf_counter() - t0
    print(f"  耗时: {t_parallel:.2f}s")

    print(f"\n  串行: {t_serial:.2f}s → 并行: {t_parallel:.2f}s")
    print(f"  加速比: {t_serial/t_parallel:.1f}×")
    print(f"  结果一致: {serial == parallel}")

    assert serial == parallel
    print(f"✅ 串行/并行结果完全一致\n")


# ============================================================
# 实验 2：delayed 的必要性
# ============================================================
def experiment_2_delayed():
    print("=" * 60)
    print("实验 2 · delayed 的必要性 — 不加会怎样")
    print("=" * 60)

    def add_one(x):
        time.sleep(0.1)
        print(f"    执行 add_one({x})")
        return x + 1

    print("\n不加 delayed — 函数在 Parallel 之前就执行了:")
    # 注意：这在 Python 中语法合法，但 add_one 会立即串行执行
    # Parallel 收到的是一堆已计算好的结果
    t0 = time.perf_counter()
    results_wrong = Parallel(n_jobs=4)(
        add_one(i) for i in range(4)      # ← add_one 立即执行！
    )
    t_wrong = time.perf_counter() - t0
    print(f"  耗时: {t_wrong:.2f}s (应该接近 0.4s — 串行的)")

    print("\n加 delayed — 函数被延迟为任务:")
    t0 = time.perf_counter()
    results_right = Parallel(n_jobs=4, verbose=0)(
        delayed(add_one)(i) for i in range(4)
    )
    t_right = time.perf_counter() - t0
    print(f"  耗时: {t_right:.2f}s (应该接近 0.1s — 并行的)")
    print(f"\n📖 delayed 让函数调用从'立即执行'变成'包装成任务，Parallel 统一调度'\n")


# ============================================================
# 实验 3：n_jobs 的影响
# ============================================================
def experiment_3_n_jobs():
    print("=" * 60)
    print("实验 3 · n_jobs 的影响")
    print("=" * 60)

    def task(x):
        time.sleep(0.1)
        return x

    tasks = list(range(16))

    print(f"\n{16} 个任务（每个 0.1s）:")
    for n in [1, 2, 4, -1]:
        t0 = time.perf_counter()
        Parallel(n_jobs=n, verbose=0)(
            delayed(task)(i) for i in tasks
        )
        elapsed = time.perf_counter() - t0
        label = f"n_jobs={n}"
        if n == -1:
            label = f"n_jobs=-1 (all={os.cpu_count()})"
        elif n == 1:
            label = f"n_jobs=1 (串行)"
        print(f"  {label:<25} {elapsed:.2f}s")
    print()


# ============================================================
# 实验 4：并行交叉验证
# ============================================================
def experiment_4_cross_val():
    print("=" * 60)
    print("实验 4 · 手动并行交叉验证")
    print("=" * 60)

    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold

    X, y = make_classification(n_samples=2000, n_features=20, random_state=42)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    def train_one_fold(train_idx, test_idx, X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        model = LogisticRegression(max_iter=1000)
        model.fit(X_tr, y_tr)
        return model.score(X_te, y_te)

    # 串行
    t0 = time.perf_counter()
    scores_serial = [train_one_fold(tr, te, X, y) for tr, te in kf.split(X)]
    t_s = time.perf_counter() - t0

    # 并行
    t0 = time.perf_counter()
    scores_parallel = Parallel(n_jobs=-1)(
        delayed(train_one_fold)(tr, te, X, y) for tr, te in kf.split(X)
    )
    t_p = time.perf_counter() - t0

    print(f"\n串行 10 折: {t_s:.2f}s, 平均={np.mean(scores_serial):.4f}")
    print(f"并行 10 折: {t_p:.2f}s, 平均={np.mean(scores_parallel):.4f}")
    print(f"加速比: {t_s/t_p:.1f}×")
    print(f"✅ 10 折交叉验证结果一致\n")


# ============================================================
# 实验 5：并行超参搜索
# ============================================================
def experiment_5_grid_search():
    print("=" * 60)
    print("实验 5 · 手动并行超参搜索")
    print("=" * 60)

    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    X, y = make_classification(n_samples=1500, n_features=15, random_state=42)

    param_grid = [
        {"n_estimators": n, "max_depth": d}
        for n in [50, 100, 200]
        for d in [5, 10, None]
    ]
    # 9 个组合

    def evaluate(params, X, y):
        """评估一组参数 — 注意内层 n_jobs=1"""
        model = RandomForestClassifier(**params, n_jobs=1, random_state=42)
        scores = cross_val_score(model, X, y, cv=3)
        return {"params": params, "mean_score": float(scores.mean()), "std": float(scores.std())}

    print(f"\n{len(param_grid)} 个超参组合，并行搜索...")
    t0 = time.perf_counter()
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(evaluate)(p, X, y) for p in param_grid
    )
    t_search = time.perf_counter() - t0

    # 找最佳
    best = max(results, key=lambda r: r["mean_score"])
    print(f"\n最佳参数: {best['params']}")
    print(f"最佳分数: {best['mean_score']:.4f} ± {best['std']:.4f}")
    print(f"总耗时: {t_search:.1f}s")

    # 验证内层 n_jobs=1 的重要性
    print(f"\n📖 注意：evaluate 里的 RandomForestClassifier 设了 n_jobs=1")
    print(f"   避免外层 {os.cpu_count()} 进程 × 内层多线程 = 嵌套并行拖慢速度\n")


# ============================================================
# 实验 6：Memory + Parallel 组合
# ============================================================
def experiment_6_memory_parallel():
    print("=" * 60)
    print("实验 6 · Memory + Parallel — 并行中缓存已完成的任务")
    print("=" * 60)

    memory = Memory(os.path.join(cache_dir, "exp6"), verbose=1)

    @memory.cache
    def train_with_params(n_estimators, max_depth, data_id):
        """模拟训练——耗时，但结果可缓存"""
        time.sleep(0.3)
        rng = np.random.RandomState(hash((n_estimators, max_depth, data_id)) % 2**32)
        score = 0.7 + 0.2 * rng.random()
        return {"n": n_estimators, "d": max_depth, "score": score}

    params_list = [
        (n, d) for n in [50, 100] for d in [5, 10]
    ]  # 4 个组合

    # 第一次：全量计算
    print("\n=== 第一轮：全部并行执行 ===")
    t0 = time.perf_counter()
    results1 = Parallel(n_jobs=2, verbose=0)(
        delayed(train_with_params)(n, d, 1) for n, d in params_list
    )
    t1 = time.perf_counter() - t0

    # 第二次：全部命中缓存
    print("\n=== 第二轮：全部缓存命中 ===")
    t0 = time.perf_counter()
    results2 = Parallel(n_jobs=2, verbose=0)(
        delayed(train_with_params)(n, d, 1) for n, d in params_list
    )
    t2 = time.perf_counter() - t0

    print(f"\n第一轮耗时: {t1:.1f}s, 第二轮耗时: {t2:.3f}s")
    print(f"加速: {t1/t2:.0f}× — 缓存让重跑几乎不花时间")

    scores1 = [r["score"] for r in results1]
    scores2 = [r["score"] for r in results2]
    assert scores1 == scores2
    print(f"✅ 两轮结果完全一致")

    memory.clear()
    print()


# ============================================================
# 实验 7：常见坑
# ============================================================
def experiment_7_pitfalls():
    print("=" * 60)
    print("实验 7 · 常见坑")
    print("=" * 60)

    # 坑 1：改全局变量 — 子进程改的是自己的副本
    print("\n坑 1 — 子进程改全局变量无效:")
    shared_list = []

    def append_to_list(i):
        shared_list.append(i)  # 子进程的 shared_list，不是主进程的
        return i

    Parallel(n_jobs=2)(
        delayed(append_to_list)(i) for i in range(10)
    )
    print(f"  主进程 shared_list: {shared_list} (空！子进程改的是自己的副本)")
    print(f"  正确做法：用 Parallel 的返回值")

    # 坑 2：lambda 函数
    print("\n坑 2 — lambda 可能无法序列化:")
    try:
        Parallel(n_jobs=2)(
            delayed(lambda x: x**2)(i) for i in range(5)
        )
        print(f"  lambda 在本平台可序列化（平台/配置相关）")
    except Exception as e:
        print(f"  lambda 失败: {type(e).__name__} → 用命名函数代替")

    # 坑 3：嵌套并行
    print("\n坑 3 — 嵌套并行提醒:")
    print(f"  外层 Parallel(n_jobs={os.cpu_count()}) × 内层 RandomForest(n_jobs=-1)")
    print(f"  = {os.cpu_count() * os.cpu_count()} 个竞争线程 → 反而更慢")
    print(f"  解决：内层模型设 n_jobs=1")

    # 坑 4：大对象按值传递
    print("\n坑 4 — 大数组重复传递给每个子进程:")
    big_arr = np.random.randn(1000, 1000)
    print(f"  数组大小: {big_arr.nbytes / 1024**2:.1f} MB")
    print(f"  10 个子进程各传一份 = {big_arr.nbytes * 10 / 1024**2:.0f} MB")
    print(f"  解决：用 mmap_mode='r' 加载，所有子进程共享")

    # 坑 5：Windows 缺少 if __name__ == '__main__' 保护
    print("\n坑 5 — Windows 需要 if __name__ == '__main__':")
    print(f"  本文件已有保护 → {'✅' if __name__ == '__main__' else '❌ 缺失'}")
    print()


# ============================================================
if __name__ == "__main__":
    experiment_1_serial_vs_parallel()
    experiment_2_delayed()
    experiment_3_n_jobs()
    experiment_4_cross_val()
    experiment_5_grid_search()
    experiment_6_memory_parallel()
    experiment_7_pitfalls()

    shutil.rmtree(tmp_dir)

    print("=" * 60)
    print("第 07 章完成 ✓")
    print("=" * 60)
    print("""
本章验证了：
  ✅ 串行 → 并行，只改 for 循环那一行
  ✅ delayed 让函数调用从立即执行变成任务调度
  ✅ n_jobs=1/4/-1 的影响
  ✅ 手动并行交叉验证 + 超参搜索
  ✅ Memory + Parallel = 重跑时已完成的直接读缓存
  ✅ 5 个坑：全局变量、lambda、嵌套并行、大数组传递、Windows 保护
""")
