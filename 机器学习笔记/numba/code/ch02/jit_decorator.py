"""
第二章：JIT 装饰器 —— 配套代码
==================================
学习目标：
  1. 掌握 @njit / @jit 的各种用法
  2. 理解函数签名的声明方式
  3. 熟悉编译选项（cache, fastmath, inline, boundscheck）
  4. 学会使用 inspect_* 查看编译结果
"""
import time
import numpy as np
from numba import njit, jit, float64, int64, boolean, void
from numba.types import float64 as f8, int64 as i8

# ──────────────────────────────────────────────
# 示例 1: @njit vs @jit 的行为差异
# ──────────────────────────────────────────────

@njit
def nopython_loop(arr):
    """强制 nopython：全部编译为机器码"""
    s = 0.0
    for x in arr:
        s += x
    return s


@jit  # 允许回退到 object 模式
def mixed_allowed(arr):
    """允许回退：能用 njit 就 njit，不行就 object"""
    s = 0.0
    for x in arr:
        s += x
    return s


def demo_njit_vs_jit():
    print("=" * 55)
    print("示例 1: @njit vs @jit")
    print("=" * 55)

    arr = np.arange(1000, dtype=np.float64)

    # @njit
    t0 = time.perf_counter()
    r1 = nopython_loop(arr)
    t1 = time.perf_counter() - t0
    print(f"  @njit 结果: {r1:.1f}, 首次编译耗时: {t1*1000:.3f}ms")

    # @jit (行为类似，因为没有不可编译的内容)
    t0 = time.perf_counter()
    r2 = mixed_allowed(arr)
    t2 = time.perf_counter() - t0
    print(f"  @jit  结果: {r2:.1f}, 首次编译耗时: {t2*1000:.3f}ms")

    # 查看两者的已编译签名
    print(f"  @njit 的签名列表: {nopython_loop.signatures}")
    print(f"  @jit  的签名列表: {mixed_allowed.signatures}")
    print()
    print("  ⚠️ 实践中永远用 @njit，避免静默回退到 object 模式")


# ──────────────────────────────────────────────
# 示例 2: 显示声明函数签名
# ──────────────────────────────────────────────

@njit(float64(float64, float64))
def typed_add(a, b):
    return a + b


@njit(boolean(int64[:]))
def has_positive(arr):
    """检查数组中是否有正数"""
    for x in arr:
        if x > 0:
            return True
    return False


@njit(float64[:](float64[:], float64))
def scale_array(arr, factor):
    """返回缩放后的新数组"""
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        result[i] = arr[i] * factor
    return result


@njit(void(int64))
def print_stars(n):
    """无返回值的函数"""
    for i in range(min(n, 20)):
        print("*", end="")
    print()


def demo_explicit_signatures():
    print(f"\n{'='*55}")
    print("示例 2: 显式函数签名")
    print(f"{'='*55}")

    print(f"  typed_add(3.0, 4.0) = {typed_add(3.0, 4.0)}")
    # 传 int 也会被强制转为 float64
    print(f"  typed_add(3, 4)     = {typed_add(3, 4)}  ← int 被 cast 为 float64")

    print(f"  has_positive([-1, -2, 3]) = {has_positive(np.array([-1, -2, 3]))}")
    print(f"  has_positive([-1, -2, -3]) = {has_positive(np.array([-1, -2, -3]))}")

    arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    print(f"  scale_array([1,2,3], 10) = {scale_array(arr, 10.0)}")

    print(f"  print_stars(10): ", end="")
    print_stars(10)


# ──────────────────────────────────────────────
# 示例 3: 数组签名 —— 维度、布局
# ──────────────────────────────────────────────

@njit(float64(float64[:]))
def sum_1d(arr):
    """一维数组求和"""
    s = 0.0
    for x in arr:
        s += x
    return s


@njit(float64(float64[:, :]))
def sum_2d(arr):
    """二维数组求和"""
    s = 0.0
    n, m = arr.shape
    for i in range(n):
        for j in range(m):
            s += arr[i, j]
    return s


@njit(float64(float64[::1]))     # C-contiguous 一维
def sum_c_contiguous(arr):
    """只接受 C-contiguous 的一维数组"""
    s = 0.0
    for x in arr:
        s += x
    return s


def demo_array_signatures():
    print(f"\n{'='*55}")
    print("示例 3: 数组签名")
    print(f"{'='*55}")

    arr1d = np.random.randn(1000)
    arr2d = np.random.randn(10, 100)

    print(f"  sum_1d: {sum_1d(arr1d):.4f}")
    print(f"  sum_2d: {sum_2d(arr2d):.4f}")
    print(f"  sum_c_contiguous: {sum_c_contiguous(arr1d):.4f}")

    # 尝试传 F-contiguous 数组给 sum_c_contiguous
    arr_f = np.asfortranarray(np.random.randn(1000))
    try:
        result = sum_c_contiguous(arr_f)
        print(f"  sum_c_contiguous (F-array): {result:.4f} ← 注意：Numba 自动做了拷贝")
    except Exception as e:
        print(f"  sum_c_contiguous (F-array) 失败: {e}")


# ──────────────────────────────────────────────
# 示例 4: cache=True —— 编译缓存
# ──────────────────────────────────────────────

@njit(cache=True)
def cached_square(x):
    """开启缓存后，编译结果保存到磁盘"""
    return x * x


def demo_cache():
    print(f"\n{'='*55}")
    print("示例 4: 编译缓存 cache=True")
    print(f"{'='*55}")

    print(f"  cached_square(5.0) = {cached_square(5.0)}")
    print(f"  签名列表: {cached_square.signatures}")
    print()
    print("  💡 下次运行此脚本时，该函数的编译结果")
    print("     会从 __pycache__/*.nbc 直接加载，无需重复编译")
    print("  缓存文件位置：")
    import os
    for root, dirs, files in os.walk("."):
        for f in files:
            if f.endswith(".nbc"):           # nbc = Numba byte code
                print(f"      {os.path.join(root, f)}")


# ──────────────────────────────────────────────
# 示例 5: fastmath=True —— 浮点优化
# ──────────────────────────────────────────────

@njit
def normal_sqrt_sum(arr):
    s = 0.0
    for x in arr:
        s += np.sqrt(x)
    return s


@njit(fastmath=True)
def fast_sqrt_sum(arr):
    s = 0.0
    for x in arr:
        s += np.sqrt(x)
    return s


def demo_fastmath():
    print(f"\n{'='*55}")
    print("示例 5: fastmath 浮点优化")
    print(f"{'='*55}")

    arr = np.random.rand(1_000_000).astype(np.float64)

    # 预热
    normal_sqrt_sum(arr[:100])
    fast_sqrt_sum(arr[:100])

    # Benchmark
    t0 = time.perf_counter()
    r1 = normal_sqrt_sum(arr)
    t1 = time.perf_counter() - t0

    t0 = time.perf_counter()
    r2 = fast_sqrt_sum(arr)
    t2 = time.perf_counter() - t0

    print(f"  normal sqrt: {t1*1000:.3f}ms, result={r1:.2f}")
    print(f"  fast   sqrt: {t2*1000:.3f}ms, result={r2:.2f}")
    print(f"  加速比: {t1/t2:.2f}×")
    print(f"  精度差异: {abs(r1 - r2):.2e}")

    print()
    print("  ⚠️ fastmath 的使用场景：")
    print("    ✅ 可以放宽精度要求的场景（图形、游戏、近似计算）")
    print("    ❌ 严格的科学计算、金融计算（需要 IEEE 754 保证）")


# ──────────────────────────────────────────────
# 示例 6: inline='always' —— 函数内联
# ──────────────────────────────────────────────

@njit(inline='always')
def cubic(x):
    """被频繁调用的小函数 —— 适合内联"""
    return x * x * x


@njit
def compute_poly(arr):
    """主函数：循环中调用 cubic（已被内联，无 call overhead）"""
    result = np.empty_like(arr)
    for i in range(len(arr)):
        result[i] = cubic(arr[i]) + 2 * arr[i] + 1
    return result


def demo_inline():
    print(f"\n{'='*55}")
    print("示例 6: 函数内联 inline='always'")
    print(f"{'='*55}")

    arr = np.arange(1000, dtype=np.float64)

    t0 = time.perf_counter()
    r = compute_poly(arr)
    t = time.perf_counter() - t0

    print(f"  计算 1000 个值的多项式: {t*1e6:.1f}μs")
    print(f"  cubic 函数被内联到 compute_poly 中，无调用开销")


# ──────────────────────────────────────────────
# 示例 7: nogil=True —— 释放 GIL
# ──────────────────────────────────────────────

@njit(nogil=True)
def heavy_compute(n):
    """纯数值计算 + nogil → 运行时无 GIL → 可真正多线程并行"""
    s = 0.0
    for i in range(n):
        s += np.sin(i) * np.cos(i)
    return s


def demo_nogil():
    print(f"\n{'='*55}")
    print("示例 7: nogil=True")
    print(f"{'='*55}")

    n = 1_000_000

    t0 = time.perf_counter()
    r = heavy_compute(n)
    t = time.perf_counter() - t0

    print(f"  heavy_compute({n:,}): {r:.4f}, 耗时 {t*1000:.2f}ms")
    print(f"  执行时未持有 GIL → 可在 ThreadPoolExecutor 中真正并行")


# ──────────────────────────────────────────────
# 示例 8: 编译内省 —— 查看生成的代码
# ──────────────────────────────────────────────

@njit
def add_and_double(x, y):
    return (x + y) * 2


def demo_introspection():
    print(f"\n{'='*55}")
    print("示例 8: 编译内省 inspect_*")
    print(f"{'='*55}")

    # 触发编译
    add_and_double(1, 2)

    # 查看签名
    print(f"  签名: {add_and_double.signatures}")

    # 查看 LLVM IR（关键片段）
    llvm_ir = add_and_double.inspect_llvm()
    lines = llvm_ir.get((int64, int64), '')
    line_count = len(lines.split('\n'))
    print(f"  LLVM IR: {line_count} 行")

    # 查看汇编（关键片段）
    asm = add_and_double.inspect_asm()
    asm_lines = len(asm.get((int64, int64), '').split('\n'))
    print(f"  汇编: {asm_lines} 行")

    print()
    print("  💡 inspect_types() 可查看类型推断的详细结果")
    print("  💡 inspect_llvm()  可查看生成的 LLVM IR（便于性能分析）")
    print("  💡 inspect_asm()   可查看最终的汇编指令")


# ──────────────────────────────────────────────
# 示例 9: 多签名 —— 一个函数支持多种类型
# ──────────────────────────────────────────────

@njit([float64(float64, float64),
       int64(int64, int64)])
def typed_max(a, b):
    """支持 float64 和 int64 两种类型的 max"""
    if a > b:
        return a
    return b


def demo_multiple_signatures():
    print(f"\n{'='*55}")
    print("示例 9: 多签名")
    print(f"{'='*55}")

    print(f"  typed_max(3.0, 4.0) = {typed_max(3.0, 4.0)}")
    print(f"  typed_max(3, 4)     = {typed_max(3, 4)}")
    print(f"  编译的签名列表: {typed_max.signatures}")


# ──────────────────────────────────────────────
# 示例 10: boundscheck —— 数组越界检查
# ──────────────────────────────────────────────

@njit(boundscheck=True)
def safe_access(arr, idx):
    """开发时开启 boundscheck：越界报错"""
    return arr[idx]


@njit(boundscheck=False)
def fast_access(arr, idx):
    """生产时关闭 boundscheck：无检查、更快但危险"""
    return arr[idx]


def demo_boundscheck():
    print(f"\n{'='*55}")
    print("示例 10: boundscheck")
    print(f"{'='*55}")

    arr = np.array([1, 2, 3])

    print("  boundscheck=True  (开发): ", end="")
    try:
        print(safe_access(arr, 10))
    except IndexError as e:
        print(f"IndexError → 立即发现越界 bug")

    print("  boundscheck=False (生产): ", end="")
    print("越界访问 → 未定义行为（可能 crash/读到脏数据）⚠️")


# ──────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("╔═══════════════════════════════════════════╗")
    print("║    Numba 第二章：JIT 装饰器              ║")
    print("║    配套代码演示                          ║")
    print("╚═══════════════════════════════════════════╝")

    demo_njit_vs_jit()
    demo_explicit_signatures()
    demo_array_signatures()
    demo_cache()
    demo_fastmath()
    demo_inline()
    demo_nogil()
    demo_introspection()
    demo_multiple_signatures()
    demo_boundscheck()

    print(f"\n{'='*55}")
    print("✅ 第二章代码演示完成！")
    print(f"{'='*55}")
