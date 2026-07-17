"""
第七章：@vectorize 和 @guvectorize —— 配套代码
==================================================
学习目标：
  1. 用 @vectorize 把标量函数变成 ufunc
  2. 用 @guvectorize 处理子数组操作（窗口、行列）
  3. 对比 @njit vs @vectorize vs @guvectorize
"""
import numpy as np
import time
from numba import vectorize, guvectorize, njit, float64, int64

# ══════════════════════════════════════════════
# 示例 1: @vectorize 基本用法 —— 对标量写，自动套循环
# ══════════════════════════════════════════════

@vectorize
def relu(x):
    """ReLU 激活函数：对标量写的，但可以接受数组"""
    return max(0, x)


@vectorize('float64(float64)')
def swish(x):
    """Swish 激活函数（Google Brain, 2017）"""
    return x / (1 + np.exp(-x))


@vectorize('float64(float64, float64, float64)')
def clamp(x, lo, hi):
    """三参数版本：裁剪到 [lo, hi]"""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def demo_basic_vectorize():
    print("=" * 55)
    print("示例 1: @vectorize 基本用法")
    print("=" * 55)

    arr = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    print(f"  arr    = {arr}")
    print(f"  ReLU   = {relu(arr)}")
    print(f"  Swish  = {swish(arr)}")
    print(f"  clamp  = {clamp(arr, 0.0, 1.0)}")
    print()
    print("  💡 你只写了对标量的操作，@vectorize 自动套循环")


# ══════════════════════════════════════════════
# 示例 2: @vectorize 的广播
# ══════════════════════════════════════════════

@vectorize('float64(float64, float64)')
def weighted_add(x, y):
    return x * 0.7 + y * 0.3


def demo_broadcasting():
    print(f"\n{'='*55}")
    print("示例 2: ufunc 广播")
    print(f"{'='*55}")

    # 和 NumPy ufunc 一样，自动广播
    arr1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    scalar = 10.0

    print(f"  arr   = {arr1d}")
    print(f"  scalar = {scalar}")
    print(f"  weighted_add(arr, scalar) = {weighted_add(arr1d, scalar)}")
    print(f"  weighted_add(scalar, arr) = {weighted_add(scalar, arr1d)}")
    print()
    print("  💡 @vectorize 函数天然支持 NumPy 广播规则")


# ══════════════════════════════════════════════
# 示例 3: target='parallel' —— 多核加速
# ══════════════════════════════════════════════

@vectorize('float64(float64)', target='cpu')
def cpu_heavy(x):
    s = 0.0
    for _ in range(1000):
        s += np.sin(x) * np.cos(x)
    return s


@vectorize('float64(float64)', target='parallel')
def parallel_heavy(x):
    s = 0.0
    for _ in range(1000):
        s += np.sin(x) * np.cos(x)
    return s


def demo_parallel_target():
    print(f"\n{'='*55}")
    print("示例 3: target='parallel' 多核加速")
    print(f"{'='*55}")

    arr = np.random.randn(20000).astype(np.float64)

    # 热身
    cpu_heavy(arr[:10])
    parallel_heavy(arr[:10])

    t0 = time.perf_counter()
    r1 = cpu_heavy(arr)
    t1 = time.perf_counter() - t0
    print(f"  target='cpu' (单核):      {t1:.4f}s")

    t0 = time.perf_counter()
    r2 = parallel_heavy(arr)
    t2 = time.perf_counter() - t0
    print(f"  target='parallel' (多核): {t2:.4f}s  ← 快 {t1/t2:.1f}×")
    print()
    print("  💡 target='parallel' 一键启用多核，无需手写 prange")


# ══════════════════════════════════════════════
# 示例 4: @vectorize 多签名 —— 不同类型自动分发
# ══════════════════════════════════════════════

@vectorize([float64(float64, float64),
            int64(int64, int64)])
def safe_add(a, b):
    """接受 float64 或 int64，各自编译一个版本"""
    return a + b


def demo_multiple_signatures():
    print(f"\n{'='*55}")
    print("示例 4: 多签名 —— 自动类型分发")
    print(f"{'='*55}")

    f_arr = np.array([1.5, 2.5, 3.5])
    i_arr = np.array([1, 2, 3], dtype=np.int64)

    print(f"  safe_add(float64): {safe_add(f_arr, f_arr)}")
    print(f"  safe_add(int64):   {safe_add(i_arr, i_arr)}")
    print()
    print("  💡 不同输入类型自动匹配对应的编译版本")


# ══════════════════════════════════════════════
# 示例 5: @guvectorize —— 子数组操作
# ══════════════════════════════════════════════

@guvectorize('(n) -> ()',
             'float64[:], float64[:]')
def row_mean(row, out):
    """输入一行，输出一个标量（这行的均值）"""
    s = 0.0
    for x in row:
        s += x
    out[0] = s / len(row)


@guvectorize('(n), (n) -> ()',
             'float64[:], float64[:], float64[:]')
def cosine_similarity(a, b, out):
    """两个向量 → 余弦相似度"""
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for i in range(len(a)):
        dot += a[i] * b[i]
        norm_a += a[i] * a[i]
        norm_b += b[i] * b[i]
    out[0] = dot / (np.sqrt(norm_a) * np.sqrt(norm_b) + 1e-10)


def demo_guvectorize():
    print(f"\n{'='*55}")
    print("示例 5: @guvectorize 子数组操作")
    print(f"{'='*55}")

    mat = np.array([[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0]])
    print(f"  矩阵:\n{mat}")
    print(f"  每行均值: {row_mean(mat)}")

    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    print(f"\n  a = {a}, b = {b}")
    print(f"  余弦相似度: {cosine_similarity(a, b)}")


# ══════════════════════════════════════════════
# 示例 6: @guvectorize —— 滑动窗口
# ══════════════════════════════════════════════

@guvectorize('(n), () -> (n)',
             'float64[:], int64, float64[:]')
def moving_average(x, window, out):
    """滑动窗口均值 —— 不能逐元素做，需要看邻居"""
    n = len(x)
    half = window // 2
    for i in range(n):
        total = 0.0
        count = 0
        start = max(0, i - half)
        end = min(n, i + half + 1)
        for j in range(start, end):
            total += x[j]
            count += 1
        out[i] = total / count


def demo_moving_average():
    print(f"\n{'='*55}")
    print("示例 6: @guvectorize —— 滑动窗口均值")
    print(f"{'='*55}")

    signal = np.array([1.0, 2.0, 100.0, 3.0, 2.0, 1.0, 2.0, 100.0, 1.0, 2.0])
    smoothed = moving_average(signal, 3)

    print(f"  原始信号:  {signal}")
    print(f"  平滑后:    {np.round(smoothed, 2)}")
    print(f"  (窗口大小=3, 尖峰 100 被平滑了)")
    print()
    print("  💡 这种操作需要看邻居元素，@vectorize 做不了")
    print("      @guvectorize 的 (n),()->(n) 签名让它可以")


# ══════════════════════════════════════════════
# 示例 7: @njit vs @vectorize vs @guvectorize 对比
# ══════════════════════════════════════════════

@njit
def add_njit(a, b):
    result = np.empty_like(a)
    for i in range(len(a)):
        result[i] = a[i] + b[i]
    return result


@vectorize('float64(float64, float64)')
def add_vec(a, b):
    return a + b


@guvectorize('(n), (n) -> (n)',
             'float64[:], float64[:], float64[:]')
def add_guvec(a, b, out):
    for i in range(len(a)):
        out[i] = a[i] + b[i]


def demo_comparison():
    print(f"\n{'='*55}")
    print("示例 7: @njit vs @vectorize vs @guvectorize")
    print(f"{'='*55}")

    a = np.random.randn(1_000_000).astype(np.float64)
    b = np.random.randn(1_000_000).astype(np.float64)

    # 热身
    add_njit(a[:10], b[:10])
    add_vec(a[:10], b[:10])
    out = np.empty(10)
    add_guvec(a[:10], b[:10], out)

    # @njit
    t0 = time.perf_counter()
    r1 = add_njit(a, b)
    t1 = time.perf_counter() - t0

    # @vectorize
    t0 = time.perf_counter()
    r2 = add_vec(a, b)
    t2 = time.perf_counter() - t0

    # @guvectorize
    out_full = np.empty_like(a)
    t0 = time.perf_counter()
    add_guvec(a, b, out_full)
    t3 = time.perf_counter() - t0

    print(f"  @njit:        {t1*1000:.3f}ms")
    print(f"  @vectorize:   {t2*1000:.3f}ms")
    print(f"  @guvectorize: {t3*1000:.3f}ms")
    print()
    print("  💡 对逐元素加法，三者性能接近")
    print("     选哪个取决于需求：")
    print("     - 逐元素简洁 → @vectorize")
    print("     - 子数组操作 → @guvectorize")
    print("     - 复杂多步骤 → @njit")


# ══════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("╔═══════════════════════════════════════════╗")
    print("║   Numba 第七章：@vectorize / @guvectorize ║")
    print("║   配套代码演示                           ║")
    print("╚═══════════════════════════════════════════╝")

    demo_basic_vectorize()
    demo_broadcasting()
    demo_parallel_target()
    demo_multiple_signatures()
    demo_guvectorize()
    demo_moving_average()
    demo_comparison()

    print(f"\n{'='*55}")
    print("✅ 第七章代码演示完成！")
    print(f"{'='*55}")
