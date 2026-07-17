"""
第八章：并行计算 —— 配套代码
===================================
学习目标：
  1. 用 parallel=True 启用自动并行
  2. 用 prange 显式并行循环
  3. 理解归约操作和竞态条件
  4. 测量实际加速比
"""
import numpy as np
import time
from numba import njit, prange, set_num_threads, get_num_threads
from concurrent.futures import ThreadPoolExecutor

# ══════════════════════════════════════════════
# 示例 1: parallel=True —— 自动并行化
# ══════════════════════════════════════════════

@njit
def serial_sum(arr):
    """串行版本"""
    total = 0.0
    for i in range(len(arr)):
        total += arr[i]
    return total


@njit(parallel=True)
def parallel_sum(arr):
    """并行版本：只加一个参数"""
    total = 0.0
    for i in range(len(arr)):
        total += arr[i]
    return total


def demo_auto_parallel():
    print("=" * 55)
    print("示例 1: parallel=True 自动并行化")
    print("=" * 55)
    print(f"  CPU 核心数: {get_num_threads()}")

    arr = np.random.randn(10_000_000).astype(np.float64)

    # 热身
    serial_sum(arr[:100])
    parallel_sum(arr[:100])

    t0 = time.perf_counter()
    r1 = serial_sum(arr)
    t1 = time.perf_counter() - t0
    print(f"  串行:  {t1:.4f}s, result={r1:.2f}")

    t0 = time.perf_counter()
    r2 = parallel_sum(arr)
    t2 = time.perf_counter() - t0
    print(f"  并行:  {t2:.4f}s, result={r2:.2f}  ← 快 {t1/t2:.1f}×")
    print()
    print("  💡 只改了一行：@njit → @njit(parallel=True)")


# ══════════════════════════════════════════════
# 示例 2: prange —— 显式并行循环
# ══════════════════════════════════════════════

@njit
def element_wise_serial(arr):
    """串行：用 range"""
    result = np.empty_like(arr)
    for i in range(len(arr)):
        result[i] = np.sin(arr[i]) * np.cos(arr[i])
    return result


@njit(parallel=True)
def element_wise_parallel(arr):
    """并行：用 prange"""
    result = np.empty_like(arr)
    for i in prange(len(arr)):         # ← 只改这一个词
        result[i] = np.sin(arr[i]) * np.cos(arr[i])
    return result


def demo_prange():
    print(f"\n{'='*55}")
    print("示例 2: prange —— 显式并行循环")
    print(f"{'='*55}")

    arr = np.random.randn(5_000_000).astype(np.float64)

    # 热身
    element_wise_serial(arr[:100])
    element_wise_parallel(arr[:100])

    t0 = time.perf_counter()
    r1 = element_wise_serial(arr)
    t1 = time.perf_counter() - t0
    print(f"  range:  {t1:.4f}s")

    t0 = time.perf_counter()
    r2 = element_wise_parallel(arr)
    t2 = time.perf_counter() - t0
    print(f"  prange: {t2:.4f}s  ← 快 {t1/t2:.1f}×")

    print()
    print("  💡 range → prange，每次迭代必须独立（不能依赖其他迭代）")


# ══════════════════════════════════════════════
# 示例 3: 归约操作 —— Numba 自动处理
# ══════════════════════════════════════════════

@njit(parallel=True)
def reductions(arr):
    """多种归约操作"""
    # 求和
    total = 0.0
    for i in prange(len(arr)):
        total += arr[i]

    # 求最大值
    max_val = arr[0]
    for i in prange(len(arr)):
        if arr[i] > max_val:
            max_val = arr[i]

    # 计数
    count = 0
    for i in prange(len(arr)):
        if arr[i] > 0:
            count += 1

    return total, max_val, count


def demo_reductions():
    print(f"\n{'='*55}")
    print("示例 3: 归约操作")
    print(f"{'='*55}")

    arr = np.random.randn(1_000_000).astype(np.float64)
    total, max_val, count = reductions(arr)

    print(f"  数组大小: {len(arr)}")
    print(f"  总和:     {total:.2f}")
    print(f"  最大值:   {max_val:.4f}")
    print(f"  正数个数: {count}")
    print()
    print("  💡 归约操作 (sum/max/count) Numba 自动做线程合并")


# ══════════════════════════════════════════════
# 示例 4: 矩阵行操作 —— 行之间独立，可并行
# ══════════════════════════════════════════════

@njit
def row_normalize_serial(mat):
    """串行：每行归一化"""
    m, n = mat.shape
    result = np.empty_like(mat)
    for i in range(m):
        row_sum = 0.0
        for j in range(n):
            row_sum += mat[i, j]
        for j in range(n):
            result[i, j] = mat[i, j] / (row_sum + 1e-10)
    return result


@njit(parallel=True)
def row_normalize_parallel(mat):
    """并行：外层用 prange，内层用 range"""
    m, n = mat.shape
    result = np.empty_like(mat)
    for i in prange(m):                # 行之间独立 → 可并行
        row_sum = 0.0
        for j in range(n):             # 列要顺序访问（内存连续）
            row_sum += mat[i, j]
        for j in range(n):
            result[i, j] = mat[i, j] / (row_sum + 1e-10)
    return result


def demo_row_operations():
    print(f"\n{'='*55}")
    print("示例 4: 矩阵行操作")
    print(f"{'='*55}")

    mat = np.random.randn(10000, 100).astype(np.float64)

    # 热身
    row_normalize_serial(mat[:10])
    row_normalize_parallel(mat[:10])

    t0 = time.perf_counter()
    r1 = row_normalize_serial(mat)
    t1 = time.perf_counter() - t0
    print(f"  串行:  {t1:.4f}s")

    t0 = time.perf_counter()
    r2 = row_normalize_parallel(mat)
    t2 = time.perf_counter() - t0
    print(f"  并行:  {t2:.4f}s  ← 快 {t1/t2:.1f}×")

    print()
    print("  💡 外层 prange (行并行), 内层 range (列顺序)")


# ══════════════════════════════════════════════
# 示例 5: 竞态条件演示
# ══════════════════════════════════════════════

@njit(parallel=True)
def safe_write(arr):
    """✅ 安全：每个迭代写不同位置"""
    result = np.empty_like(arr)
    for i in prange(len(arr)):
        result[i] = arr[i] * 2     # 每个线程写 result[i]，互不干扰
    return result


@njit(parallel=True)
def unsafe_write_demo(arr):
    """⚠️ 演示：多个迭代可能写同一位置 (简化示例)"""
    # 实际场景：histogram、分组聚合等
    # 这里用简化示例说明概念
    buckets = np.zeros(10)
    for i in prange(len(arr)):
        bucket_idx = int(arr[i]) % 10
        buckets[bucket_idx] += 1   # 多个线程可能同时写 buckets[bucket_idx]
    return buckets


def demo_race_condition():
    print(f"\n{'='*55}")
    print("示例 5: 竞态条件")
    print(f"{'='*55}")

    arr = np.random.randint(0, 100, size=100000)

    # ✅ 安全的写入
    safe = safe_write(arr.astype(np.float64))
    print(f"  ✅ 安全写入：每个 i 写 result[i]，OK")

    # ⚠️ 不安全的写入
    # 多次运行可能结果不同（线程调度不确定）
    unsafe1 = unsafe_write_demo(arr)
    unsafe2 = unsafe_write_demo(arr)
    print(f"  ⚠️ 不安全写入 (多线程写同一位置):")
    print(f"     第1次: {unsafe1[:5]}")
    print(f"     第2次: {unsafe2[:5]}")
    if not np.array_equal(unsafe1, unsafe2):
        print(f"     两次结果不同！← 竞态条件")
    else:
        print(f"     两次碰巧相同（但不保证每次都相同）")

    print()
    print("  💡 避免竞态：确保每次迭代写不同位置")
    print("     或用 Numba 支持的归约模式 (sum/max/min)")


# ══════════════════════════════════════════════
# 示例 6: 加速比测量 —— 理想 vs 实际
# ══════════════════════════════════════════════

@njit
def compute_intensive_serial(n):
    """计算密集型任务（串行）"""
    result = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(1000):
            s += np.sin(i * 0.001) * np.cos(j * 0.001)
        result[i] = s
    return result


@njit(parallel=True)
def compute_intensive_parallel(n):
    """计算密集型任务（并行）"""
    result = np.zeros(n)
    for i in prange(n):
        s = 0.0
        for j in range(1000):
            s += np.sin(i * 0.001) * np.cos(j * 0.001)
        result[i] = s
    return result


def demo_speedup():
    print(f"\n{'='*55}")
    print("示例 6: 加速比测量")
    print(f"{'='*55}")

    n_cores = get_num_threads()
    print(f"  CPU 核心数: {n_cores}")

    n = 10000

    # 热身
    compute_intensive_serial(10)
    compute_intensive_parallel(10)

    t0 = time.perf_counter()
    r1 = compute_intensive_serial(n)
    t_serial = time.perf_counter() - t0
    print(f"  串行:  {t_serial:.4f}s")

    t0 = time.perf_counter()
    r2 = compute_intensive_parallel(n)
    t_parallel = time.perf_counter() - t0
    print(f"  并行:  {t_parallel:.4f}s")

    speedup = t_serial / t_parallel
    efficiency = speedup / n_cores

    print()
    print(f"  实际加速比: {speedup:.2f}×")
    print(f"  理想加速比: {n_cores}× (核心数)")
    print(f"  并行效率:   {efficiency*100:.1f}%  (实际/理想)")

    print()
    print("  💡 实际加速比 ≈ 0.6~0.8 × 核心数是正常的")
    print("     原因：线程开销、内存带宽、缓存竞争")


# ══════════════════════════════════════════════
# 示例 7: 线程数控制
# ══════════════════════════════════════════════

def demo_thread_control():
    print(f"\n{'='*55}")
    print("示例 7: 线程数控制")
    print(f"{'='*55}")

    arr = np.random.randn(1_000_000).astype(np.float64)

    # 测试不同线程数
    original_threads = get_num_threads()
    print(f"  默认线程数: {original_threads}")

    for n_threads in [1, 2, 4, original_threads]:
        set_num_threads(n_threads)

        # 热身
        parallel_sum(arr[:100])

        t0 = time.perf_counter()
        result = parallel_sum(arr)
        t = time.perf_counter() - t0
        print(f"  {n_threads} 线程: {t:.4f}s")

    # 恢复原始线程数
    set_num_threads(original_threads)

    print()
    print("  💡 大部分情况用默认线程数就好")
    print("     除非要和其他并行任务共享 CPU")


# ══════════════════════════════════════════════
# 示例 8: nogil=True —— 配合 Python 多线程
# ══════════════════════════════════════════════

@njit
def worker_with_gil(data):
    """❌ 默认：执行时持有 GIL"""
    return np.sum(data ** 2)


@njit(nogil=True)
def worker_no_gil(data):
    """✅ nogil=True：释放 GIL"""
    return np.sum(data ** 2)


def demo_nogil():
    print(f"\n{'='*55}")
    print("示例 8: nogil=True —— 配合 Python 多线程")
    print(f"{'='*55}")

    # 准备 4 个独立任务
    tasks = [np.random.randn(2_000_000).astype(np.float64) for _ in range(4)]

    # 热身
    worker_with_gil(tasks[0][:100])
    worker_no_gil(tasks[0][:100])

    # ❌ 默认版本（有 GIL）：Python 多线程串行执行
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as executor:
        results1 = list(executor.map(worker_with_gil, tasks))
    t_with_gil = time.perf_counter() - t0
    print(f"  ❌ 默认 (持有 GIL):   {t_with_gil:.4f}s")
    print(f"     → 4个线程因 GIL 而排队，实际串行")

    # ✅ nogil=True 版本：真正并行
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as executor:
        results2 = list(executor.map(worker_no_gil, tasks))
    t_no_gil = time.perf_counter() - t0
    print(f"  ✅ nogil=True:        {t_no_gil:.4f}s  ← 快 {t_with_gil/t_no_gil:.1f}×")
    print(f"     → 4个线程真正并行")

    print()
    print("  💡 nogil=True 适合：")
    print("     - 多个独立任务，用 Python ThreadPoolExecutor 调度")
    print("     - 配合 concurrent.futures / threading")
    print()
    print("  💡 parallel=True 适合：")
    print("     - 单个大数组，Numba 内部自动并行")
    print("     - 不想自己管理线程")


# ══════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("╔═══════════════════════════════════════════╗")
    print("║    Numba 第八章：并行计算                 ║")
    print("║    配套代码演示                          ║")
    print("╚═══════════════════════════════════════════╝")

    demo_auto_parallel()
    demo_prange()
    demo_reductions()
    demo_row_operations()
    demo_race_condition()
    demo_speedup()
    demo_thread_control()
    demo_nogil()

    print(f"\n{'='*55}")
    print("✅ 第八章代码演示完成！")
    print(f"{'='*55}")
