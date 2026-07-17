"""
第一章：Numba 概述与安装 —— 配套代码
==========================================
学习目标：
  1. 亲手体验 Numba 的加速效果
  2. 理解"预热"（首次编译）开销
  3. 对比纯 Python / NumPy / Numba 的循环性能
"""
import time
import numpy as np
from numba import njit, jit

np.random.seed(42)

# ──────────────────────────────────────────────
# 示例 1: 最简单的 Numba 加速 —— 数组求和
# ──────────────────────────────────────────────

def sum_pure_python(arr):
    """纯 Python 循环求和（基线）"""
    total = 0.0
    for x in arr:
        total += x
    return total


def sum_numpy(arr):
    """NumPy 内置 sum"""
    return np.sum(arr)


@njit
def sum_numba(arr):
    """Numba 加速的循环求和"""
    total = 0.0
    for x in arr:
        total += x
    return total


@njit(fastmath=True)
def sum_numba_fastmath(arr):
    """Numba + fastmath 浮点优化"""
    total = 0.0
    for x in arr:
        total += x
    return total


def benchmark_sum(n=10_000_000):
    """对比四种求和方法"""
    arr = np.random.randn(n).astype(np.float64)

    print(f"\n{'='*60}")
    print(f"基准测试：数组求和 (n={n:,})")
    print(f"{'='*60}")

    # 纯 Python 循环（只跑一轮因为太慢）
    if n <= 10_000_000:
        t0 = time.perf_counter()
        result_py = sum_pure_python(arr)
        t_py = time.perf_counter() - t0
        print(f"  纯 Python 循环: {t_py:.4f}s  (太慢了！)")

    # NumPy
    t0 = time.perf_counter()
    result_np = sum_numpy(arr)
    t_np = time.perf_counter() - t0
    print(f"  NumPy sum:      {t_np:.4f}s")

    # Numba（包含首次编译时间）
    t0 = time.perf_counter()
    result_nb = sum_numba(arr)
    t_nb = time.perf_counter() - t0
    print(f"  Numba @njit:     {t_nb:.4f}s  (含首次编译)")

    # Numba（第二次调用，已预热）
    t0 = time.perf_counter()
    result_nb2 = sum_numba(arr)
    t_nb2 = time.perf_counter() - t0
    print(f"  Numba @njit:     {t_nb2:.4f}s  (已预热)")

    # Numba + fastmath（已预热）
    t0 = time.perf_counter()
    result_nbf = sum_numba_fastmath(arr)
    t_nbf = time.perf_counter() - t0
    print(f"  Numba +fastmath: {t_nbf:.4f}s")


# ──────────────────────────────────────────────
# 示例 2: 复杂循环 —— 不能简单向量化的场景
# ──────────────────────────────────────────────

def fibonacci_python(n):
    """计算 Fibonacci 数列第 n 项（循环版）"""
    if n < 2:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


@njit
def fibonacci_numba(n):
    """同样的代码，Numba 加速"""
    if n < 2:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def benchmark_fibonacci():
    print(f"\n{'='*60}")
    print("基准测试：Fibonacci 数列 (n=1,000,000)")
    print(f"{'='*60}")

    n = 1_000_000

    # 纯 Python
    t0 = time.perf_counter()
    fib_py = fibonacci_python(n)
    t_py = time.perf_counter() - t0
    print(f"  纯 Python:     {t_py:.4f}s")

    # Numba（含编译）
    t0 = time.perf_counter()
    fib_nb = fibonacci_numba(n)
    t_nb = time.perf_counter() - t0
    print(f"  Numba (首次):   {t_nb:.4f}s  (含编译)")

    # Numba（预热后）
    t0 = time.perf_counter()
    fib_nb2 = fibonacci_numba(n)
    t_nb2 = time.perf_counter() - t0
    print(f"  Numba (预热):   {t_nb2:.4f}s")


# ──────────────────────────────────────────────
# 示例 3: 多重签名 —— 不同类型触发重新编译
# ──────────────────────────────────────────────

@njit
def multiply(a, b):
    """同一个函数，不同类型参数会编译多个版本"""
    return a * b


def demo_multiple_signatures():
    print(f"\n{'='*60}")
    print("演示：不同类型触发多次编译")
    print(f"{'='*60}")

    # 第一次：int64 版本
    print("  调用 multiply(3, 4) →", multiply(3, 4))
    print("  → 编译 multiply(int64, int64)")

    # 第二次：float64 版本（触发重新编译）
    print("  调用 multiply(3.0, 4.0) →", multiply(3.0, 4.0))
    print("  → 编译 multiply(float64, float64)")

    # 第三次：同类型，复用
    print("  调用 multiply(5, 6) →", multiply(5, 6))
    print("  → 复用 multiply(int64, int64)，不再编译")


# ──────────────────────────────────────────────
# 示例 4: 显式指定函数签名
# ──────────────────────────────────────────────

from numba import float64, int64

@njit(float64(float64, float64))
def add_floats(a, b):
    """签名强制 a 和 b 是 float64，返回 float64"""
    return a + b


def demo_explicit_signature():
    print(f"\n{'='*60}")
    print("演示：显式函数签名")
    print(f"{'='*60}")
    print("  签名: float64(float64, float64)")
    print("  add_floats(3.14, 2.72) →", add_floats(3.14, 2.72))
    # 即使传 int，也会被强制转为 float64
    print("  add_floats(3, 4) →", add_floats(3, 4), "  (int 被转为 float64)")


# ──────────────────────────────────────────────
# 示例 5: 编译缓存演示（cache=True）
# ──────────────────────────────────────────────

@njit(cache=True)
def cached_function(a, b):
    """开启 cache 后，编译结果持久化到磁盘"""
    result = 0.0
    for i in range(1000):
        result += a * b
    return result


def demo_cache():
    print(f"\n{'='*60}")
    print("演示：编译缓存 (cache=True)")
    print(f"{'='*60}")
    print("  首次调用 → 编译并写入磁盘缓存")
    t0 = time.perf_counter()
    r = cached_function(3.0, 4.0)
    t = time.perf_counter() - t0
    print(f"  结果: {r}, 耗时: {t:.4f}s")
    print("  第二次调用 → 直接从缓存加载")
    print("  缓存位置: __pycache__ 目录下的 .nbc 文件")


# ──────────────────────────────────────────────
# 示例 6: 查看 Numba 类型推断结果
# ──────────────────────────────────────────────

from numba import typeof

def demo_type_inference():
    print(f"\n{'='*60}")
    print("演示：Numba 类型反射")
    print(f"{'='*60}")

    arr = np.array([1.0, 2.0, 3.0])
    print(f"  typeof(arr) = {typeof(arr)}")
    print(f"    → arr 是 float64 的一维 C-contiguous 数组")

    arr2d = np.ones((3, 4), dtype=np.int32, order='F')
    print(f"  typeof(arr2d) = {typeof(arr2d)}")
    print(f"    → arr2d 是 int32 的二维 F-contiguous 数组")


# ──────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("╔═══════════════════════════════════════════╗")
    print("║      Numba 第一章：概述与安装             ║")
    print("║      配套代码演示                         ║")
    print("╚═══════════════════════════════════════════╝")

    # 跑演示（不跑大型 benchmark）
    demo_multiple_signatures()
    demo_explicit_signature()
    demo_cache()
    demo_type_inference()

    # 性能 benchmark（可能需要几秒）
    benchmark_fibonacci()

    # 大型 benchmark，建议运行时取消注释
    print(f"\n{'='*60}")
    print("提示：大规模 benchmark (n=10M) 需要 ~1-2秒")
    print("      取消 benchmark_sum() 的注释来运行")
    print(f"{'='*60}")
    # benchmark_sum(n=10_000_000)

    print("\n✅ 第一章代码演示完成！")
