"""
第六章：性能优化策略 —— 配套代码
======================================
学习目标：
  1. 亲眼看到每种优化手段的实际加速效果
  2. 学会正确的基准测试方法
  3. 建立"先 profile 再优化"的意识
"""
import numpy as np
import time
from numba import njit, typeof

# ══════════════════════════════════════════════
# 示例 1: 循环内分配 vs 循环外分配
# ══════════════════════════════════════════════

@njit
def bad_alloc_inside(arr):
    """❌ 每次循环迭代都分配新数组"""
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        temp = np.zeros(100)               # 每次分配 100 个 float64
        result[i] = np.sum(temp) * arr[i]
    return result


@njit
def good_alloc_outside(arr):
    """✅ 在循环外分配一次"""
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    temp = np.zeros(100)                   # 只分配一次
    for i in range(n):
        result[i] = np.sum(temp) * arr[i]
    return result


def demo_alloc():
    print("=" * 55)
    print("示例 1: 循环内分配 vs 循环外分配")
    print("=" * 55)

    n = 50000
    arr = np.random.randn(n).astype(np.float64)

    # 热身
    bad_alloc_inside(arr[:10])
    good_alloc_outside(arr[:10])

    t0 = time.perf_counter()
    bad_alloc_inside(arr)
    t1 = time.perf_counter() - t0
    print(f"  ❌ 循环内分配: {t1:.4f}s")

    t0 = time.perf_counter()
    good_alloc_outside(arr)
    t2 = time.perf_counter() - t0
    print(f"  ✅ 循环外分配: {t2:.4f}s  ← 快 {t1/t2:.1f}×")
    print("  原因: malloc 比加法贵 100-1000 倍")


# ══════════════════════════════════════════════
# 示例 2: 反复读数组元素 vs 局部变量缓存
# ══════════════════════════════════════════════

@njit
def slow_multiple_read(arr):
    """❌ 重复从内存读 arr[i]"""
    s = 0.0
    for i in range(len(arr)):
        s += arr[i] * arr[i] + arr[i] * 2.0 + arr[i]
    return s


@njit
def fast_local_variable(arr):
    """✅ 一次读取，局部变量复用"""
    s = 0.0
    for i in range(len(arr)):
        x = arr[i]                         # 只读一次
        s += x * x + x * 2.0 + x           # 局部变量直接用
    return s


def demo_local_variable():
    print(f"\n{'='*55}")
    print("示例 2: 反复读取 vs 局部变量缓存")
    print(f"{'='*55}")

    arr = np.random.randn(5_000_000).astype(np.float64)

    # 热身
    slow_multiple_read(arr[:100])
    fast_local_variable(arr[:100])

    t0 = time.perf_counter()
    r1 = slow_multiple_read(arr)
    t1 = time.perf_counter() - t0
    print(f"  ❌ 多次读 arr[i]: {t1:.4f}s, result={r1:.2f}")

    t0 = time.perf_counter()
    r2 = fast_local_variable(arr)
    t2 = time.perf_counter() - t0
    print(f"  ✅ 局部变量缓存:   {t2:.4f}s, result={r2:.2f}  ← 快 {t1/t2:.1f}×")
    print("  原因: 寄存器访问 ~0 周期，内存访问 ~100 周期")


# ══════════════════════════════════════════════
# 示例 3: 分开的循环 vs 循环融合
# ══════════════════════════════════════════════

@njit
def separate_passes(arr):
    """❌ 三次独立的数据遍历 + 两个中间数组"""
    a = arr + 1.0              # pass 1: 遍历 + 分配临时数组 a
    b = a * 2.0                # pass 2: 遍历 + 分配临时数组 b
    c = np.sqrt(b)             # pass 3: 遍历 + 分配结果 c
    return c


@njit
def fused_pass(arr):
    """✅ 一个循环，零中间数组"""
    n = len(arr)
    c = np.empty_like(arr)
    for i in range(n):
        tmp = arr[i] + 1.0
        tmp = tmp * 2.0
        c[i] = np.sqrt(tmp)
    return c


def demo_fusion():
    print(f"\n{'='*55}")
    print("示例 3: 独立循环 vs 循环融合")
    print(f"{'='*55}")

    arr = np.random.randn(5_000_000).astype(np.float64)

    # 热身
    separate_passes(arr[:100])
    fused_pass(arr[:100])

    t0 = time.perf_counter()
    r1 = separate_passes(arr)
    t1 = time.perf_counter() - t0
    print(f"  ❌ 3 次独立遍历: {t1:.4f}s")

    t0 = time.perf_counter()
    r2 = fused_pass(arr)
    t2 = time.perf_counter() - t0
    print(f"  ✅ 1 次融合遍历:   {t2:.4f}s  ← 快 {t1/t2:.1f}×")
    print(f"  中间数组节省: ~{5_000_000*8*2/1024/1024:.0f} MB")
    print("  原因: 减少 2/3 数据遍历 + 省掉 2 个临时数组")


# ══════════════════════════════════════════════
# 示例 4: 遍历方向 —— 匹配内存布局
# ══════════════════════════════════════════════

@njit
def traverse_row_first(arr):
    """✅ 外层行、内层列 —— C 数组友好"""
    n, m = arr.shape
    s = 0.0
    for i in range(n):
        for j in range(m):
            s += arr[i, j]
    return s


@njit
def traverse_col_first(arr):
    """❌ 外层列、内层行 —— C 数组不友好，跳跃访问"""
    n, m = arr.shape
    s = 0.0
    for j in range(m):
        for i in range(n):
            s += arr[i, j]
    return s


def demo_traversal():
    print(f"\n{'='*55}")
    print("示例 4: 遍历方向 vs 内存布局")
    print(f"{'='*55}")

    n, m = 3000, 3000
    arr = np.random.randn(n, m).astype(np.float64)     # C-contiguous

    # 热身
    traverse_row_first(arr[:50, :50])
    traverse_col_first(arr[:50, :50])

    t0 = time.perf_counter()
    r1 = traverse_row_first(arr)
    t1 = time.perf_counter() - t0
    print(f"  ✅ 逐行优先 (内层列): {t1:.4f}s  ← 内存连续")

    t0 = time.perf_counter()
    r2 = traverse_col_first(arr)
    t2 = time.perf_counter() - t0
    print(f"  ❌ 逐列优先 (内层行): {t2:.4f}s  ← 慢 {t2/t1:.1f}×，内存跳跃")

    print(f"  typeof(arr): {typeof(arr)}")
    print("  原因: C 数组的行元素在内存中紧挨着")


# ══════════════════════════════════════════════
# 示例 5: 预分配输出数组 (out parameter)
# ══════════════════════════════════════════════

@njit
def compute_with_alloc(a, b):
    """❌ 每次调用都分配结果数组"""
    return a * a + b * b


@njit
def compute_no_alloc(a, b, out):
    """✅ 结果写入预分配的 out 数组"""
    for i in range(len(a)):
        out[i] = a[i] * a[i] + b[i] * b[i]


def demo_prealloc():
    print(f"\n{'='*55}")
    print("示例 5: 内部分配 vs 预分配输出 (out parameter)")
    print(f"{'='*55}")

    a = np.random.randn(1_000_000).astype(np.float64)
    b = np.random.randn(1_000_000).astype(np.float64)
    out = np.empty_like(a)

    # 热身
    compute_with_alloc(a[:10], b[:10])
    compute_no_alloc(a[:10], b[:10], out[:10])

    # 模拟 1000 次循环（类似 ML 训练中的 epoch）
    t0 = time.perf_counter()
    for _ in range(100):
        _ = compute_with_alloc(a, b)          # 每次分配 8MB
    t1 = time.perf_counter() - t0
    print(f"  ❌ 每次内部分配: {t1:.4f}s (100 次调用，每次分配 8MB)")

    t0 = time.perf_counter()
    for _ in range(100):
        compute_no_alloc(a, b, out)           # 0 次分配
    t2 = time.perf_counter() - t0
    print(f"  ✅ 外部预分配:   {t2:.4f}s (100 次调用，0 次分配)  ← 快 {t1/t2:.1f}×")
    print("  适合: 被循环反复调用的函数 (ML 训练、数值模拟)")


# ══════════════════════════════════════════════
# 示例 6: 正确 vs 错误的基准测试
# ══════════════════════════════════════════════

@njit
def heavy_loop(arr):
    """用于演示的耗时函数"""
    s = 0.0
    for x in arr:
        s += np.sin(x) * np.cos(x)
    return s


def demo_benchmark():
    print(f"\n{'='*55}")
    print("示例 6: 正确的基准测试方法")
    print(f"{'='*55}")

    arr = np.random.randn(1_000_000).astype(np.float64)

    # ❌ 错误做法：包含编译时间
    t0 = time.perf_counter()
    r1 = heavy_loop(arr)
    t_bad = time.perf_counter() - t0
    print(f"  ❌ 首次调用(含编译): {t_bad:.4f}s")
    print(f"     → 结果是 {r1:.2f}，但时间不是真实的运行时间")

    # ✅ 正确做法：热身后再计时
    heavy_loop(arr[:10])                       # 热身
    t0 = time.perf_counter()
    for _ in range(10):
        heavy_loop(arr)
    t_good = (time.perf_counter() - t0) / 10
    print(f"  ✅ 热身后的平均:     {t_good:.4f}s")
    print(f"     → 编译被排除，这才是真实性能")

    print()
    print("  黄金法则: 永远用 dummy 数据跑一次再计时")
    print("  或用 timeit 模块代替手写计时")


# ══════════════════════════════════════════════
# 示例 7: Amdahl 定律演示
# ══════════════════════════════════════════════

def simulate_amdahl():
    """演示 Amdahl 定律：为什么不能给所有代码加 @njit"""
    print(f"\n{'='*55}")
    print("示例 7: 加速比、热点 和 Amdahl 定律")
    print(f"{'='*55}")

    print("""
  ┌─────────────────────────────────────────────────────┐
  │ 1. 什么是"加速比"？                                  │
  │    加速比 = 优化前时间 / 优化后时间                    │
  │    例：原来 10s → 优化后 2s → 加速比 = 5×            │
  ├─────────────────────────────────────────────────────┤
  │ 2. 什么是"热点"？                                    │
  │    程序里占运行时间最多的那部分代码                     │
  │    例：总运行 100s，compute() 占了 80s → 热点        │
  ├─────────────────────────────────────────────────────┤
  │ 3. Amdahl 定律核心公式：                              │
  │    加速比 = 1 / ((1-f) + f/s)                       │
  │    f = 热点占比   s = 对热点加速的倍数               │
  └─────────────────────────────────────────────────────┘
  """)

    # 计算不同场景下的加速比
    scenarios = [
        # (描述,     f热点占比,  s热点加速倍数)
        ("热点占90%, 热点加速10×  (优化对了)",  0.90, 10),
        ("热点占90%, 热点加速50×  (优化对了)",  0.90, 50),
        ("热点占10%, 热点加速10×  (优化错了!)", 0.10, 10),
        ("热点占10%, 热点加速100× (优化错了!)", 0.10, 100),
        ("热点占50%, 热点加速50× ",            0.50, 50),
    ]

    for desc, f, s in scenarios:
        speedup = 1 / ((1 - f) + f / s)
        print(f"  {desc}")
        print(f"    加速比 = 1/({1-f:.2f} + {f:.2f}/{s}) = {speedup:.1f}×")
        if speedup < 2:
            print(f"    ⚠️ 加速比 < 2×，不值得费劲！")
        print()

    print("  💡 教训：")
    print("     1. 先找到热点（占总时间最多的代码），只优化热点")
    print("     2. 占 10% 的代码即使加速 100×，整体也只快 1.1×")
    print("     3. 占 90% 的代码加速 10×，整体就能快 5×")
    print()
    print("  💡 怎么找热点：")
    print("     python -m cProfile -s cumtime your_script.py")
    print("     看 tottime 和 cumtime 列，最大的就是热点")


# ══════════════════════════════════════════════
# 示例 8: 用 cProfile 找热点 —— 可运行演示

def slow_compute(n):
    """模拟一个热点函数"""
    data = np.random.randn(n)
    result = 0.0
    for i in range(len(data)):
        for j in range(100):
            result += np.sin(data[i]) * np.cos(j * 0.01)
    return result


def demo_profile():
    """用 cProfile 找到热点"""
    print(f"\n{'='*55}")
    print("示例 8: cProfile 找热点")
    print(f"{'='*55}")

    import cProfile
    import pstats
    import io

    profiler = cProfile.Profile()
    profiler.enable()
    result = slow_compute(2000)
    profiler.disable()

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumtime')
    ps.print_stats(10)

    print("  cProfile 输出（按 cumtime 排序，前 10 名）：")
    for line in s.getvalue().split('\n')[:12]:
        print(f"  {line}")

    print(f"\n  解读：")
    print(f"    ncalls  = 函数被调用次数")
    print(f"    tottime = 函数自身耗时（不含子函数）")
    print(f"    cumtime = 函数总耗时（含子函数）→ 找热点看这个！")


# 示例 9: 减少分支 —— 无分支写法
# ══════════════════════════════════════════════

@njit
def branch_version(arr):
    """❌ 循环内 if-else"""
    n = len(arr)
    r = np.empty_like(arr)
    for i in range(n):
        if arr[i] > 0:
            r[i] = np.sqrt(arr[i])
        else:
            r[i] = -np.sqrt(-arr[i])
    return r


@njit
def branchless_version(arr):
    """✅ 用符号位代替 if-else"""
    signs = np.sign(arr)                    # -1, 0, 1
    return signs * np.sqrt(np.abs(arr))


def demo_branch_removal():
    print(f"{'='*55}")
    print("示例 9: 分支 vs 无分支写法")
    print(f"{'='*55}")

    arr = np.random.randn(1_000_000).astype(np.float64)

    # 热身
    branch_version(arr[:100])
    branchless_version(arr[:100])

    t0 = time.perf_counter()
    r1 = branch_version(arr)
    t1 = time.perf_counter() - t0
    print(f"  ❌ 循环内 if-else: {t1:.4f}s")

    t0 = time.perf_counter()
    r2 = branchless_version(arr)
    t2 = time.perf_counter() - t0
    print(f"  ✅ np.sign 无分支:  {t2:.4f}s  ← 快 {t1/t2:.1f}×")

    # 验证结果一致
    diff = np.max(np.abs(r1 - r2))
    print(f"  最大误差: {diff:.2e}  (浮点舍入差异)")

    print("  💡 分支预测失败代价高，能用数学操作避免 if 就避免")
    print("     但不是总能做到 —— 复杂的条件逻辑就保留 if")


# ══════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("╔═══════════════════════════════════════════╗")
    print("║    Numba 第六章：性能优化策略           ║")
    print("║    配套代码演示 (含基准测试)           ║")
    print("╚═══════════════════════════════════════════╝")

    demo_alloc()
    demo_local_variable()
    demo_fusion()
    demo_traversal()
    demo_prealloc()
    demo_benchmark()
    simulate_amdahl()
    demo_profile()
    demo_branch_removal()

    print(f"\n{'='*55}")
    print("✅ 第六章代码演示完成！")
    print(f"{'='*55}")
