"""
第 08 章 · 并行后端选型 — 验证代码

目的：对比 loky vs threading 在不同场景下的表现差异。

实验：
  1. CPU 密集 — loky vs threading
  2. I/O 密集 — loky vs threading
  3. numpy 运算 — threading 也可以（C 扩展释放 GIL）
  4. parallel_backend() 上下文管理器
  5. threading 共享大对象 — 零序列化开销
  6. 用 threading 加速 sklearn 的 n_jobs

运行方式：
    uv run python code/08-并行后端选型.py
"""

import os
import sys
import time
import tempfile
import shutil

import numpy as np
from joblib import Parallel, delayed, parallel_backend

# ============================================================
tmp_dir = tempfile.mkdtemp(prefix="joblib_ch08_")
print(f"Python: {sys.version.split()[0]}, CPU: {os.cpu_count()} 核\n")


# ============================================================
# 实验 1：CPU 密集 — loky 胜出
# ============================================================
def experiment_1_cpu_bound():
    print("=" * 60)
    print("实验 1 · CPU 密集 — loky vs threading")
    print("=" * 60)

    def cpu_heavy(i):
        """纯 Python 计算——GIL 的典型受害者"""
        total = 0
        for j in range(5_000_000):
            total += j % 7
        return total

    n_tasks = 8

    # 串行基线
    t0 = time.perf_counter()
    [cpu_heavy(i) for i in range(n_tasks)]
    t_serial = time.perf_counter() - t0

    # loky
    t0 = time.perf_counter()
    Parallel(n_jobs=4, backend='loky', verbose=0)(
        delayed(cpu_heavy)(i) for i in range(n_tasks)
    )
    t_loky = time.perf_counter() - t0

    # threading
    t0 = time.perf_counter()
    Parallel(n_jobs=4, backend='threading', verbose=0)(
        delayed(cpu_heavy)(i) for i in range(n_tasks)
    )
    t_thread = time.perf_counter() - t0

    print(f"\n  纯 Python 循环 × {n_tasks} 任务:")
    print(f"    串行:      {t_serial:.2f}s")
    print(f"    loky:      {t_loky:.2f}s (加速 {t_serial/t_loky:.1f}×)")
    print(f"    threading: {t_thread:.2f}s (加速 {t_serial/t_thread:.1f}×)")
    print(f"\n📖 loky 真正并行，threading 受 GIL 限制和串行差不多\n")


# ============================================================
# 实验 2：I/O 密集 — threading 胜出
# ============================================================
def experiment_2_io_bound():
    print("=" * 60)
    print("实验 2 · I/O 密集 — loky vs threading")
    print("=" * 60)

    # 模拟 I/O：写临时文件
    def io_task(i):
        time.sleep(0.1)           # 模拟 I/O 等待
        path = os.path.join(tmp_dir, f"io_{i}.txt")
        with open(path, "w") as f:
            f.write(f"data_{i}" * 1000)
        return i

    n_tasks = 16

    # 串行
    t0 = time.perf_counter()
    [io_task(i) for i in range(n_tasks)]
    t_serial = time.perf_counter() - t0

    # loky
    t0 = time.perf_counter()
    Parallel(n_jobs=4, backend='loky', verbose=0)(
        delayed(io_task)(i) for i in range(n_tasks)
    )
    t_loky = time.perf_counter() - t0

    # threading
    t0 = time.perf_counter()
    Parallel(n_jobs=4, backend='threading', verbose=0)(
        delayed(io_task)(i) for i in range(n_tasks)
    )
    t_thread = time.perf_counter() - t0

    print(f"\n  I/O 任务 (sleep + 写文件) × {n_tasks}:")
    print(f"    串行:      {t_serial:.2f}s")
    print(f"    loky:      {t_loky:.2f}s (加速 {t_serial/t_loky:.1f}×)")
    print(f"    threading: {t_thread:.2f}s (加速 {t_serial/t_thread:.1f}×)")
    print(f"\n📖 I/O 密集型任务，threading 和 loky 都能加速")
    print(f"   但 threading 没有进程启动开销，通常略快。\n")


# ============================================================
# 实验 3：numpy 运算 — threading 也可以
# ============================================================
def experiment_3_numpy():
    print("=" * 60)
    print("实验 3 · numpy 运算 — threading 也能并行")
    print("=" * 60)

    # numpy 的 C 扩展在执行时会释放 GIL
    # 所以 numpy 运算用 threading 也能真正并行
    def numpy_heavy(i):
        a = np.random.randn(800, 800)
        b = np.random.randn(800, 800)
        return float(np.dot(a, b).trace())

    n_tasks = 8

    t0 = time.perf_counter()
    [numpy_heavy(i) for i in range(n_tasks)]
    t_serial = time.perf_counter() - t0

    t0 = time.perf_counter()
    Parallel(n_jobs=4, backend='loky', verbose=0)(
        delayed(numpy_heavy)(i) for i in range(n_tasks)
    )
    t_loky = time.perf_counter() - t0

    t0 = time.perf_counter()
    Parallel(n_jobs=4, backend='threading', verbose=0)(
        delayed(numpy_heavy)(i) for i in range(n_tasks)
    )
    t_thread = time.perf_counter() - t0

    print(f"\n  numpy dot product 800×800 × {n_tasks}:")
    print(f"    串行:      {t_serial:.2f}s")
    print(f"    loky:      {t_loky:.2f}s (加速 {t_serial/t_loky:.1f}×)")
    print(f"    threading: {t_thread:.2f}s (加速 {t_serial/t_thread:.1f}×)")
    print(f"\n📖 numpy 的 C 扩展释放 GIL → threading 也能真正并行\n")


# ============================================================
# 实验 4：parallel_backend 上下文管理器
# ============================================================
def experiment_4_context():
    print("=" * 60)
    print("实验 4 · parallel_backend() 上下文管理器")
    print("=" * 60)

    def task(x):
        return x

    # 默认后端
    with parallel_backend('loky', n_jobs=2):
        r1 = Parallel()(delayed(task)(i) for i in range(4))
        print(f"  loky 上下文内:      {r1}")

    # 切换后端
    with parallel_backend('threading', n_jobs=2):
        r2 = Parallel()(delayed(task)(i) for i in range(4))
        print(f"  threading 上下文内: {r2}")

    # 退出后恢复默认
    r3 = Parallel(n_jobs=2)(delayed(task)(i) for i in range(4))
    print(f"  上下文外（默认）:    {r3}")

    print(f"\n📖 用 parallel_backend() 可以在不修改 Parallel 调用的情况下切换后端")
    print(f"   特别适合控制 sklearn 内部的并行行为\n")


# ============================================================
# 实验 5：使用 parallel_backend 改变 sklearn 的并行
# ============================================================
def experiment_5_sklearn_backend():
    print("=" * 60)
    print("实验 5 · 用 parallel_backend 控制 sklearn 内部并行")
    print("=" * 60)

    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier

    X, y = make_classification(n_samples=2000, n_features=30, random_state=42)

    # 默认：sklearn 内部用 loky
    t0 = time.perf_counter()
    rf1 = RandomForestClassifier(n_estimators=100, n_jobs=4, random_state=42)
    rf1.fit(X, y)
    t_default = time.perf_counter() - t0

    # 用 parallel_backend 改成 threading
    with parallel_backend('threading'):
        t0 = time.perf_counter()
        rf2 = RandomForestClassifier(n_estimators=100, n_jobs=4, random_state=42)
        rf2.fit(X, y)
        t_thread = time.perf_counter() - t0

    print(f"\n  RandomForest(n_estimators=100, n_jobs=4):")
    print(f"    默认 loky:     {t_default:.2f}s")
    print(f"    threading:     {t_thread:.2f}s")
    print(f"\n📖 两种后端都能加速 sklearn，差异取决于数据和硬件")
    print(f"   通常大数据集用 loky，小数据+特征多试 threading\n")


# ============================================================
if __name__ == "__main__":
    experiment_1_cpu_bound()
    experiment_2_io_bound()
    experiment_3_numpy()
    experiment_4_context()
    experiment_5_sklearn_backend()

    shutil.rmtree(tmp_dir)

    print("=" * 60)
    print("第 08 章完成 ✓")
    print("=" * 60)
    print("""
本章验证了：
  ✅ CPU 密集 → loky 真正并行，threading 受 GIL 限制
  ✅ I/O 密集 → threading 和 loky 都能加速，threading 稍快
  ✅ numpy 运算 → threading 也能并行（C 扩展释放 GIL）
  ✅ parallel_backend() → 切换后端控制所有 Parallel
  ✅ sklearn n_jobs → 用 parallel_backend 可以改变其内部并行后端
""")
