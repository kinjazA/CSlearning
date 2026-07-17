"""
第五章：支持的 NumPy 特性 —— 配套代码
==========================================
学习目标：
  1. 确认 Numba 中哪些 NumPy 函数可用
  2. 理解 Numba 随机数和外面 NumPy 随机数的区别
  3. 学会处理不支持的 NumPy 函数
"""
import numpy as np
from numba import njit, typeof
import time

# ══════════════════════════════════════════════
# 示例 1: 数组创建 —— 对比各种创建方式
# ══════════════════════════════════════════════

@njit
def create_all(n):
    """演示所有支持的数组创建方式"""
    z = np.zeros(n, dtype=np.float64)
    o = np.ones(n, dtype=np.float64)
    e = np.empty(n, dtype=np.float64)     # 未初始化，垃圾值！
    for i in range(n):                     # ⚠️ empty 后必须手动赋值
        e[i] = float(i)
    a = np.arange(n, dtype=np.float64)
    lin = np.linspace(0, 1, n)
    eye3 = np.eye(3)
    full = np.full(n, 42.0)
    copy = np.copy(a)
    return e, lin, full, copy  # 返回几个代表性的


@njit
def create_like(arr):
    """_like 系列：创建和输入数组相同形状/类型的数组"""
    return np.zeros_like(arr), np.ones_like(arr), np.empty_like(arr)


def demo_creation():
    print("=" * 55)
    print("示例 1: 数组创建")
    print("=" * 55)

    e, lin, full, copy = create_all(5)
    print(f"  np.empty(5) 后赋值:  {e}")
    print(f"  np.linspace(0,1,5): {lin}")
    print(f"  np.full(5, 42):     {full}")
    print(f"  np.copy:            {copy}")

    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    z, o, e2 = create_like(arr)
    print(f"  zeros_like(2×2):\n{z}")
    print(f"  ones_like(2×2):\n{o}")


# ══════════════════════════════════════════════
# 示例 2: 元素运算 + 数学函数
# ══════════════════════════════════════════════

@njit
def math_operations(arr):
    """元素级运算 + 常用数学函数"""
    # 基本运算
    a = arr + 1.0
    b = arr * 2.0
    c = arr ** 3

    # 数学函数
    d = np.sqrt(np.abs(arr))      # sqrt(|x|)
    e = np.exp(-arr ** 2)         # exp(-x²)
    f = np.clip(arr, -1.0, 1.0)   # 截断
    g = np.where(arr > 0, arr, 0) # ReLU：正数保留，负数变 0
    h = np.sin(arr) ** 2 + np.cos(arr) ** 2  # 恒为 1

    return a, d, f, g, h


@njit
def logical_ops(arr, threshold):
    """比较 + 布尔运算"""
    mask1 = arr > threshold           # 比较
    mask2 = arr < -threshold
    combined = mask1 | mask2          # 用 | 而不是 or！
    count = np.sum(combined)          # True=1, False=0
    return count, combined


def demo_math():
    print(f"\n{'='*55}")
    print("示例 2: 元素运算和数学函数")
    print(f"{'='*55}")

    arr = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    a, d, f, g, h = math_operations(arr)
    print(f"  arr        = {arr}")
    print(f"  arr + 1    = {a}")
    print(f"  sqrt(|x|)  = {np.round(d, 3)}")
    print(f"  clip(-1,1) = {f}")
    print(f"  ReLU       = {g}")
    print(f"  sin²+cos²  = {np.round(h, 10)}  ← 恒为 1.0")

    cnt, mask = logical_ops(arr, 1.5)
    print(f"  |x|>1.5 的个数: {cnt}")
    print(f"  布尔 mask: {mask}")

    print()
    print("  ⚠️ Numba 中数组逻辑用 & / |，不是 and / or")


# ══════════════════════════════════════════════
# 示例 3: 聚合 + 轴操作
# ══════════════════════════════════════════════

@njit
def aggregation_demo(arr):
    """各种聚合函数"""
    s = np.sum(arr)
    m = np.mean(arr)
    std = np.std(arr)
    v = np.var(arr)
    mn = np.min(arr)
    mx = np.max(arr)
    imin = np.argmin(arr)
    imax = np.argmax(arr)
    return s, m, std, v, mn, mx, imin, imax


@njit
def axis_ops(arr):
    """按轴聚合"""
    row_sums = np.sum(arr, axis=1)   # 每行求和
    col_means = np.mean(arr, axis=0) # 每列求均值
    global_max = np.max(arr)         # 全局最大值
    return row_sums, col_means, global_max


def demo_aggregation():
    print(f"\n{'='*55}")
    print("示例 3: 聚合与轴操作")
    print(f"{'='*55}")

    arr = np.array([5.0, 2.0, 8.0, 1.0, 9.0, 3.0])
    s, m, std, v, mn, mx, imin, imax = aggregation_demo(arr)
    print(f"  arr     = {arr}")
    print(f"  sum     = {s:.2f}")
    print(f"  mean    = {m:.2f}")
    print(f"  std     = {std:.2f}")
    print(f"  min     = {mn}, 位置 = {imin}")
    print(f"  max     = {mx}, 位置 = {imax}")

    mat = np.array([[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0]])
    rows, cols, gmax = axis_ops(mat)
    print(f"\n  matrix:\n{mat}")
    print(f"  每行求和 (axis=1): {rows}")
    print(f"  每列均值 (axis=0): {cols}")


# ══════════════════════════════════════════════
# 示例 4: 数组操作 —— 索引 / reshape / 拼接
# ══════════════════════════════════════════════

@njit
def slice_demo(arr):
    """切片操作"""
    first = arr[0]
    last = arr[-1]
    middle = arr[1:4]
    every_other = arr[::2]
    return first, last, middle, every_other


@njit
def reshape_demo(arr, new_shape):
    """reshape + flatten"""
    reshaped = np.reshape(arr, new_shape)
    flat = arr.ravel()                    # 展平
    return reshaped, flat


@njit
def concat_demo(a, b):
    """数组拼接"""
    return np.concatenate((a, b))


@njit
def dot_demo(A, B):
    """矩阵乘法"""
    return np.dot(A, B)


def demo_manipulation():
    print(f"\n{'='*55}")
    print("示例 4: 数组操作")
    print(f"{'='*55}")

    arr = np.array([10, 20, 30, 40, 50, 60], dtype=np.float64)
    first, last, mid, eo = slice_demo(arr)
    print(f"  arr[0]     = {first}")
    print(f"  arr[-1]    = {last}")
    print(f"  arr[1:4]   = {mid}")
    print(f"  arr[::2]   = {eo}")

    arr2d = np.arange(6, dtype=np.float64).reshape(2, 3)
    reshaped, flat = reshape_demo(arr2d, (3, 2))
    print(f"\n  原矩阵 (2,3):\n{arr2d}")
    print(f"  reshape → (3,2):\n{reshaped}")
    print(f"  ravel →: {flat}")

    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0, 5.0])
    print(f"\n  concatenate([1,2], [3,4,5]) = {concat_demo(a, b)}")

    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0]])
    C = dot_demo(A, B)
    print(f"  np.dot(A, B):\n{C}")


# ══════════════════════════════════════════════
# 示例 5: 线性代数
# ══════════════════════════════════════════════

@njit
def solve_linear_system(A, b):
    """解 Ax = b"""
    return np.linalg.solve(A, b)


@njit
def matrix_decompositions(A):
    """多种矩阵分解"""
    detA = np.linalg.det(A)
    invA = np.linalg.inv(A)
    return detA, invA


def demo_linalg():
    print(f"\n{'='*55}")
    print("示例 5: 线性代数")
    print(f"{'='*55}")

    A = np.array([[2.0, 1.0],
                  [1.0, 3.0]], dtype=np.float64)
    b = np.array([5.0, 8.0], dtype=np.float64)

    x = solve_linear_system(A, b)
    print(f"  解 Ax = b:")
    print(f"  A = [[2,1],[1,3]], b = [5,8]")
    print(f"  x = {x}")
    print(f"  验证 A @ x = {A @ x}  ← 应该等于 [5, 8]")

    det, inv = matrix_decompositions(A)
    print(f"\n  det(A) = {det:.2f}")
    print(f"  inv(A) =\n{inv}")
    print(f"  A @ inv(A) =\n{A @ inv}  ← 应该是单位矩阵")


# ══════════════════════════════════════════════
# 示例 6: 随机数 —— Numba 的独立 RNG 系统
# ══════════════════════════════════════════════

# --- 6a: 两套 RNG 的隔离 ---

def demo_rng_isolation():
    """演示：外部 NumPy 种子和 Numba 内部种子完全隔离"""
    print(f"\n{'='*55}")
    print("示例 6a: RNG 隔离 —— 外部分部是两个世界")
    print(f"{'='*55}")

    # 外部种子
    np.random.seed(42)
    ext1 = np.random.randn(3)

    # Numba 内部（没有设种子）
    @njit
    def numba_no_seed():
        return np.random.randn(3)

    nb1 = numba_no_seed()

    print(f"  外部 np.random.seed(42) → {ext1}")
    print(f"  Numba 内部(无种子)      → {nb1}")
    print(f"  两者相同吗？ {np.allclose(ext1, nb1)}")

    # 即使在 Numba 内部设了同样的种子，结果也不同（不同算法）
    @njit
    def numba_seeded():
        np.random.seed(42)
        return np.random.randn(3)

    nb2 = numba_seeded()
    print(f"\n  外部 seed(42)      → {ext1}")
    print(f"  Numba内seed(42)    → {nb2}")
    print(f"  同种子+同代码=同结果吗？ {np.allclose(ext1, nb2)}")
    print(f"  (外部PCG64 vs Numba xoroshiro128+ → 算法不同，输出不同)")


# --- 6b: 确定性种子 ---

@njit
def deterministic_kernel(data, seed):
    """
    方案 1: 入口显式播种
    在函数第一行设种子 → 相同的 seed + 相同的代码 → 100% 可复现
    """
    np.random.seed(seed)
    noise = np.random.randn(len(data))
    return data + noise * 0.1


def demo_deterministic_seed():
    print(f"\n{'='*55}")
    print("示例 6b: 方案 1 —— 入口显式播种（确定性方案）")
    print(f"{'='*55}")

    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    r1 = deterministic_kernel(data, 42)
    r2 = deterministic_kernel(data, 42)
    r3 = deterministic_kernel(data, 99)

    print(f"  输入: {data}")
    print(f"  seed=42 第1次: {np.round(r1, 6)}")
    print(f"  seed=42 第2次: {np.round(r2, 6)}")
    print(f"  前两次相同: {np.array_equal(r1, r2)}  ← ✅ 100% 可复现")
    print(f"  seed=99 第1次: {np.round(r3, 6)}")
    print(f"  seed不同结果不同: {not np.array_equal(r1, r3)}")

    print()
    print("  📝 原理：np.random.seed() 重置当前线程的 xoroshiro128+ 状态")
    print("     同样的 seed + 同样的代码路径 → 同样的随机数序列")


# --- 6c: 并行场景下的非确定性演示 ---

@njit
def parallel_feature_build(data):
    """
    模拟并行特征构建（实际有 prange 时更复杂）
    这里展示核心概念：多个调用点各自消耗独立的随机数序列
    """
    n = len(data)
    result = np.empty_like(data)

    # 模拟两个"线程"各自生成随机数
    # 线程 A: 构建特征 A
    np.random.seed(42)
    noise_a = np.random.randn(n)
    result += noise_a * 0.5

    # 线程 B: 构建特征 B
    np.random.seed(99)
    noise_b = np.random.randn(n)
    result += noise_b * 0.3

    return result


def demo_parallel_rng():
    print(f"\n{'='*55}")
    print("示例 6c: 并行 RNG —— 每个线程独立的随机状态")
    print(f"{'='*55}")

    data = np.zeros(5)
    r1 = parallel_feature_build(data)
    r2 = parallel_feature_build(data)

    print(f"  第1次: {np.round(r1, 6)}")
    print(f"  第2次: {np.round(r2, 6)}")
    print(f"  两次相同: {np.array_equal(r1, r2)}  ← 串行下可以确定")

    print()
    print("  📝 但在 prange 多线程中：")
    print("     线程调度的非确定性 → 哪个线程先跑不确定")
    print("     → 同样的种子，不同的执行顺序 → 不同的结果!")
    print("     → 粒子滤波/随机森林中 ±0.1 RMSE 抖动")


# --- 6d: 方案 2 —— 多次运行平均 ---

@njit
def monte_carlo_step(data, seed):
    """一次 Monte Carlo 步骤"""
    np.random.seed(seed)
    noise = np.random.randn(len(data))
    return np.sum(data + noise * 0.1)


def demo_multirun_averaging():
    print(f"\n{'='*55}")
    print("示例 6d: 方案 2 —— 多次运行平均（降方差）")
    print(f"{'='*55}")

    data = np.ones(100) * 10.0
    n_runs = 20

    estimates = np.empty(n_runs)
    for i in range(n_runs):
        # 每次运行用不同的种子
        estimates[i] = monte_carlo_step(data, seed=1000 + i * 100)

    mean_est = np.mean(estimates)
    std_est = np.std(estimates)

    print(f"  运行次数: {n_runs}")
    print(f"  均值: {mean_est:.4f}")
    print(f"  标准差: {std_est:.4f}")
    print(f"  真值(无噪声): {np.sum(data):.4f}")
    print(f"  各次估计: {np.round(estimates, 4)}")

    print()
    print("  📝 原理：每次运行用不同种子 → 偏差方向随机")
    print("     N 次平均 → 偏差相互抵消 → 无偏估计")
    print("     标准差量化了不确定性的程度")


# --- 6e: 综合速查 ---

def demo_rng_quickref():
    print(f"\n{'='*55}")
    print("示例 6e: 随机数问题速查")
    print(f"{'='*55}")

    print("""
  ┌──────────────────────────────────────────┐
  │         Numba 随机数问题速查               │
  ├──────────────────────────────────────────┤
  │                                          │
  │  Q: 结果每次都不一样？                     │
  │  A: 在 @njit 函数入口加 np.random.seed()  │
  │                                          │
  │  Q: 种子设了但并行结果还是不一样？          │
  │  A: 线程调度非确定性 →                     │
  │     方案1: seed + thread_id 确定性偏移     │
  │     方案2: 跑N次取平均                     │
  │                                          │
  │  Q: 外部设的 seed 对 @njit 有效吗？        │
  │  A: 无效，两套完全独立的 RNG               │
  │                                          │
  │  Q: 论文需要可复现？                       │
  │  A: 方案1 + 记录 seed + 方案2 汇报均值±std │
  │                                          │
  └──────────────────────────────────────────┘
  """)


# ══════════════════════════════════════════════
# 示例 7: 不支持的函数 & 手写替代
# ══════════════════════════════════════════════

@njit
def manual_median(arr):
    """np.median 不支持 → 手写一个"""
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_arr[mid - 1] + sorted_arr[mid]) / 2.0
    else:
        return sorted_arr[mid]


@njit
def manual_percentile(arr, q):
    """np.percentile 不支持 → 手写一个"""
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    idx = q / 100.0 * (n - 1)
    lower = int(idx)
    upper = lower + 1
    if upper >= n:
        return sorted_arr[lower]
    frac = idx - lower
    return sorted_arr[lower] * (1 - frac) + sorted_arr[upper] * frac


@njit
def manual_unique(arr):
    """np.unique 不支持 → 手写排序+去重"""
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    if n == 0:
        return np.empty(0, dtype=arr.dtype)
    # 先分配足够大的空间
    result = np.empty(n, dtype=arr.dtype)
    result[0] = sorted_arr[0]
    j = 1
    for i in range(1, n):
        if sorted_arr[i] != sorted_arr[i - 1]:
            result[j] = sorted_arr[i]
            j += 1
    return result[:j]


def demo_manual_implementations():
    print(f"\n{'='*55}")
    print("示例 7: 不支持的函数 → 手写替代")
    print(f"{'='*55}")

    arr = np.array([5.0, 2.0, 8.0, 1.0, 2.0, 5.0, 9.0, 1.0])

    print(f"  原始数组:  {arr}")
    print(f"  排序后:    {np.sort(arr)}")
    print(f"  manual_median:     {manual_median(arr):.2f}")
    print(f"  manual_percentile(25%): {manual_percentile(arr, 25):.2f}")
    print(f"  manual_percentile(75%): {manual_percentile(arr, 75):.2f}")
    print(f"  manual_unique: {manual_unique(arr)}")

    # 验证和 NumPy 一致
    print(f"\n  验证（用外部 NumPy 对比）:")
    print(f"  np.median(arr) = {np.median(arr)}")
    print(f"  np.percentile(arr,25) = {np.percentile(arr, 25)}")
    print(f"  np.unique(arr) = {np.unique(arr)}")


# ══════════════════════════════════════════════
# 示例 8: 混合策略 —— Numba + NumPy 互补
# ══════════════════════════════════════════════

@njit
def numba_heavy_part(data):
    """Numba 处理循环密集的计算"""
    n = len(data)
    result = np.zeros(n, dtype=np.float64)
    # 复杂的逐元素迭代 —— NumPy 向量化写起来很绕
    for i in range(1, n - 1):
        if data[i] > data[i-1] and data[i] > data[i+1]:
            result[i] = data[i] * 2
        else:
            result[i] = data[i] * 0.5
    return result


def hybrid_pipeline(data):
    """混合：Numba 加速 + 外部 NumPy 做它擅长的"""
    # 第 1 步：Numba 处理复杂循环
    processed = numba_heavy_part(data)
    # 第 2 步：外部 NumPy 做统计（Numba 不支持 median）
    median_val = np.median(processed)
    mean_val = np.mean(processed)
    return processed, median_val, mean_val


def demo_hybrid():
    print(f"\n{'='*55}")
    print("示例 8: 混合策略 —— Numba + NumPy")
    print(f"{'='*55}")

    data = np.array([1.0, 5.0, 2.0, 8.0, 3.0, 10.0, 4.0])
    processed, median_val, mean_val = hybrid_pipeline(data)
    print(f"  原始数据:     {data}")
    print(f"  Numba 处理后: {processed}")
    print(f"  NumPy median: {median_val:.2f}")
    print(f"  NumPy mean:   {mean_val:.2f}")

    print()
    print("  💡 策略：把能加速的循环交给 Numba")
    print("         把 Numba 不支持的操作留给外部 NumPy")


# ══════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("╔═══════════════════════════════════════════╗")
    print("║    Numba 第五章：NumPy 特性支持           ║")
    print("║    配套代码演示                          ║")
    print("╚═══════════════════════════════════════════╝")

    demo_creation()
    demo_math()
    demo_aggregation()
    demo_manipulation()
    demo_linalg()
    demo_rng_isolation()
    demo_deterministic_seed()
    demo_parallel_rng()
    demo_multirun_averaging()
    demo_rng_quickref()
    demo_manual_implementations()
    demo_hybrid()

    print(f"\n{'='*55}")
    print("✅ 第五章代码演示完成！")
    print(f"{'='*55}")
