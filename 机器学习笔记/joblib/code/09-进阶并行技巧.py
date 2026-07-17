"""
第 09 章 · 进阶并行技巧 — 验证代码

目的：掌握 return_as、timeout、batch_size、max_nbytes、错误处理。

实验：
  1. return_as — 三种模式对比
  2. timeout — 超时跳过
  3. batch_size — 负载均衡
  4. max_nbytes — 大数组自动 mmap
  5. 错误处理 — 安全的 try/except 模式

运行方式：
    uv run python code/09-进阶并行技巧.py
"""

import os
import sys
import time
import tempfile
import shutil

import numpy as np
from joblib import Parallel, delayed

tmp_dir = tempfile.mkdtemp(prefix="joblib_ch09_")
print(f"Python: {sys.version.split()[0]}, CPU: {os.cpu_count()} 核\n")


# ============================================================
# 实验 1：return_as — 三种模式
# ============================================================
def experiment_1_return_as():
    print("=" * 60)
    print("实验 1 · return_as — list / generator / generator_unordered")
    print("=" * 60)

    def uneven_task(i):
        """耗时不均匀的任务"""
        if i < 3:
            time.sleep(0.5)
        else:
            time.sleep(0.05)
        return i

    tasks = list(range(8))

    # 模式 1：list（默认）
    print("\n1) return_as='list' (默认)— 全部算完才返回:")
    t0 = time.perf_counter()
    results = Parallel(n_jobs=4)(delayed(uneven_task)(i) for i in tasks)
    t_list = time.perf_counter() - t0
    print(f"   耗时: {t_list:.2f}s, 结果: {results}")

    # 模式 2：generator（保持顺序）
    print("\n2) return_as='generator' — 边算边返回，保持顺序:")
    t0 = time.perf_counter()
    received = []
    for r in Parallel(n_jobs=4, return_as='generator')(
        delayed(uneven_task)(i) for i in tasks
    ):
        received.append(r)
        print(f"   收到: {r} (当前已收到: {received})")
    t_gen = time.perf_counter() - t0
    print(f"   耗时: {t_gen:.2f}s, 顺序保证: {received == list(range(8))}")

    # 模式 3：generator_unordered（不保持顺序）
    print("\n3) return_as='generator_unordered' — 边算边返回，不保持顺序:")
    t0 = time.perf_counter()
    received = []
    for r in Parallel(n_jobs=4, return_as='generator_unordered')(
        delayed(uneven_task)(i) for i in tasks
    ):
        received.append(r)
        print(f"   收到: {r} (当前已收到: {received})")
    t_genu = time.perf_counter() - t0
    print(f"   耗时: {t_genu:.2f}s")
    print(f"   注意：先收到的 {received[:3]} 很可能是后面的任务（快任务先完成）")

    print(f"\n📖 generator_unordered 让你最快拿到第一批结果\n")


# ============================================================
# 实验 2：timeout — 超时
# ============================================================
def experiment_2_timeout():
    print("=" * 60)
    print("实验 2 · timeout — 防止任务卡死")
    print("=" * 60)

    def maybe_stuck(i):
        if i == 5:
            time.sleep(10)  # 模拟卡死
        else:
            time.sleep(0.1)
        return i

    print("\n设置 timeout=2s，任务 #5 会卡 10s → 超时:")
    try:
        results = Parallel(n_jobs=2, timeout=2, verbose=0)(
            delayed(maybe_stuck)(i) for i in range(8)
        )
        print(f"  全部完成: {results}")
    except Exception as e:
        print(f"  捕获异常: {type(e).__name__}")

    print(f"\n📖 timeout 让卡死的任务不会永远拖住执行")
    print(f"   如果需要'跳过'而非'全部失败'，在任务函数里用 try/except\n")


# ============================================================
# 实验 3：batch_size — 负载均衡
# ============================================================
def experiment_3_batch_size():
    print("=" * 60)
    print("实验 3 · batch_size — 控制调度粒度")
    print("=" * 60)

    def uneven(i):
        """前几个任务很慢，后面很快"""
        if i < 3:
            time.sleep(0.4)
        else:
            time.sleep(0.02)
        return i

    tasks = list(range(12))

    # batch_size='auto'（默认）
    t0 = time.perf_counter()
    Parallel(n_jobs=4)(
        delayed(uneven)(i) for i in tasks
    )
    t_auto = time.perf_counter() - t0

    # batch_size=1（最细粒度）
    t0 = time.perf_counter()
    Parallel(n_jobs=4, batch_size=1)(
        delayed(uneven)(i) for i in tasks
    )
    t_b1 = time.perf_counter() - t0

    print(f"\n  12 个任务（3慢 + 9快），4 个 worker:")
    print(f"    batch_size='auto': {t_auto:.2f}s")
    print(f"    batch_size=1:      {t_b1:.2f}s")
    print(f"\n📖 任务耗时差异大时，batch_size=1 有助于负载均衡")
    print(f"   任务耗时均匀时，默认 auto 就很好\n")


# ============================================================
# 实验 4：max_nbytes — 大数组自动 mmap
# ============================================================
def experiment_4_max_nbytes():
    print("=" * 60)
    print("实验 4 · max_nbytes — 返回大数组时自动 mmap")
    print("=" * 60)

    def return_big_array(i):
        return np.random.randn(1_000_000)  # ~8 MB

    # 默认：直接返回 ndarray
    print("\n默认行为（max_nbytes=None）:")
    results_default = Parallel(n_jobs=2)(
        delayed(return_big_array)(i) for i in range(3)
    )
    for i, r in enumerate(results_default):
        print(f"  任务{i}: type={type(r).__name__}, 内存={r.nbytes/1024**2:.0f}MB")

    # max_nbytes='1M'：大于 1MB 的数组自动转为 memmap
    print("\nmax_nbytes='1M'（大于 1MB 的自动转 memmap）:")
    results_mmap = Parallel(n_jobs=2, max_nbytes='1M')(
        delayed(return_big_array)(i) for i in range(3)
    )
    for i, r in enumerate(results_mmap):
        print(f"  任务{i}: type={type(r).__name__} (磁盘上的临时文件)")

    # 验证数据一致
    for a, b in zip(results_default, results_mmap):
        assert np.allclose(a, b)

    print(f"\n📖 100 个 8MB 结果 × 默认 = 800MB 内存")
    print(f"   100 个 8MB 结果 × max_nbytes='1M' = 几乎不占内存\n")

    del results_default, results_mmap


# ============================================================
# 实验 5：错误处理 — 安全的 try/except 模式
# ============================================================
def experiment_5_error_handling():
    print("=" * 60)
    print("实验 5 · 错误处理 — 在任务内部 try/except")
    print("=" * 60)

    def risky_division(i):
        return 100 // i  # i=0 会报错

    # ❌ 直接让 Parallel 接异常
    print("\n直接让异常飞出:")
    try:
        Parallel(n_jobs=2)(delayed(risky_division)(i) for i in range(5))
    except ZeroDivisionError:
        print(f"  任务失败 → 其他成功的结果也丢了")

    # ✅ 在任务内部处理
    print("\n任务内部 try/except:")
    def safe_division(i):
        try:
            return {"status": "ok", "i": i, "result": 100 // i}
        except Exception as e:
            return {"status": "error", "i": i, "error": str(e)}

    results = Parallel(n_jobs=2)(
        delayed(safe_division)(i) for i in range(5)
    )

    ok = [r for r in results if r["status"] == "ok"]
    err = [r for r in results if r["status"] == "error"]
    print(f"  成功: {len(ok)} 个 ({[r['i'] for r in ok]})")
    print(f"  失败: {len(err)} 个 ({[r['i'] for r in err]}) — 失败的不影响成功的")
    print(f"  成功结果: {[r['result'] for r in ok]}")

    print(f"\n📖 这是生产环境最推荐的模式：")
    print(f"   每个任务返回 {'{'}status, result, error{'}'} 字典")
    print(f"   失败的只影响自己，不影响其他任务\n")


# ============================================================
if __name__ == "__main__":
    experiment_1_return_as()
    experiment_2_timeout()
    experiment_3_batch_size()
    experiment_4_max_nbytes()
    experiment_5_error_handling()

    shutil.rmtree(tmp_dir)

    print("=" * 60)
    print("第 09 章完成 ✓")
    print("=" * 60)
    print("""
本章验证了：
  ✅ return_as — list/generator/generator_unordered 三种模式
  ✅ timeout — 防止单个任务卡死
  ✅ batch_size — 任务耗时不均时的调度优化
  ✅ max_nbytes — 返回大数组时自动 mmap 省内存
  ✅ 错误处理 — 在任务函数内 try/except，不丢成功结果
""")
