"""
第三章：Numba 类型系统 —— 配套代码
======================================
学习目标：
  1. 掌握 Numba 中的各种类型及其行为
  2. 理解数组的内存布局（C vs F）对性能的影响
  3. 学习 typed list/dict 的使用
  4. 用 typeof() 和 inspect_types() 排查类型问题
"""
import numpy as np
from numba import njit, typeof, types
from numba.typed import List, Dict

# ──────────────────────────────────────────────
# 示例 1: 类型推断 —— 看看 Numba 怎么看变量
# ──────────────────────────────────────────────

def demo_typeof():
    print("=" * 55)
    print("示例 1: numba.typeof() —— 看类型")
    print("=" * 55)

    checks = [
        ("整数 42", 42),
        ("浮点 3.14", 3.14),
        ("布尔 True", True),
        ("None", None),
        ("字符串", "hello"),
        ("1d float64 数组", np.array([1.0, 2.0, 3.0])),
        ("2d int32 C 数组", np.ones((3, 4), dtype=np.int32)),
        ("2d int32 F 数组", np.ones((3, 4), dtype=np.int32, order='F')),
        ("float32 标量", np.float32(3.14)),
        ("int8 标量", np.int8(127)),
        ("复数", 1.0 + 2.0j),
    ]
    for label, val in checks:
        print(f"  typeof({label:20s}) = {typeof(val)}")


# ──────────────────────────────────────────────
# 示例 2: 数值类型 —— 整数溢出
# ──────────────────────────────────────────────

@njit
def int_overflow():
    """演示 Numba int64 溢出（和 C 一样，不同于 Python）"""
    x = np.int64(9223372036854775807)  # int64 最大值: 2^63 - 1
    return x + 1


@njit(types.int8(types.int8, types.int8))
def add_int8(a, b):
    """显式使用 int8 —— 注意溢出的风险"""
    return a + b


def demo_numeric_types():
    print(f"\n{'='*55}")
    print("示例 2: 数值类型与溢出")
    print(f"{'='*55}")

    print("  Python int  (无限精度):")
    big = 9223372036854775807
    print(f"    9223372036854775807 + 1 = {big + 1}")

    print("  Numba int64 (会溢出):")
    print(f"    {int_overflow()}  ← 溢出回绕为 int64 最小值！")

    print("\n  int8 溢出:")
    print(f"    add_int8(100, 50) = {add_int8(100, 50)}")
    print(f"    add_int8(127, 1) = {add_int8(127, 1)}  ← 127+1 溢出了！")


# ──────────────────────────────────────────────
# 示例 3: 类型不统一 → TypingError
# ──────────────────────────────────────────────

# ❌ 这会编译失败：
# @njit
# def bad_unify(a):
#     if a > 0:
#         result = 1       # int64
#     else:
#         result = 2.0     # float64
#     return result

@njit
def good_unify(a):
    """正确：在分支前用浮点数初始化，统一类型"""
    result = 0.0           # float64 —— 先声明类型
    if a > 0:
        result = 1.0       # float64
    else:
        result = 2.0       # float64
    return result


@njit
def unify_with_cast(a):
    """也可以在分支内做类型转换"""
    if a > 0:
        result = 1         # int64
    else:
        result = 2.0       # float64
    return float(result)    # 显式转 float64


def demo_type_unification():
    print(f"\n{'='*55}")
    print("示例 3: 类型统一 (Unification)")
    print(f"{'='*55}")

    print(f"  good_unify(5)  = {good_unify(5)}")
    print(f"  good_unify(-1) = {good_unify(-1)}")

    print(f"  unify_with_cast(5)  = {unify_with_cast(5)}")
    print(f"  unify_with_cast(-1) = {unify_with_cast(-1)}")

    print()
    print("  ❌ 如果同一变量在分支中有不同类型 → TypingError")
    print("  ✅ 在分支前声明统一类型，或做显式类型转换")


# ──────────────────────────────────────────────
# 示例 4: 数组类型 —— C vs F 内存布局
# ──────────────────────────────────────────────

@njit
def sum_rows_outer(arr):
    """外层行、内层列 —— 对 C 数组友好"""
    n, m = arr.shape
    s = 0.0
    for i in range(n):
        for j in range(m):
            s += arr[i, j]
    return s


@njit
def sum_cols_outer(arr):
    """外层列、内层行 —— 对 F 数组友好"""
    n, m = arr.shape
    s = 0.0
    for j in range(m):
        for i in range(n):
            s += arr[i, j]
    return s


def demo_memory_layout_visual():
    """第一步：用一个小矩阵直观展示两种布局在内存中的差异"""
    print(f"\n{'='*55}")
    print("示例 4a: 内存布局 —— 直观演示")
    print(f"{'='*55}")

    # 小矩阵
    arr = np.array([[1,  2,  3,  4],
                    [5,  6,  7,  8],
                    [9, 10, 11, 12]], dtype=np.int32)
    print(f"  矩阵 (你看的样子):")
    print(f"  {arr}")
    print()

    # C-contiguous 的内存视图
    print(f"  行优先 (C-contiguous) —— 一行一行铺平:")
    print(f"  内存地址低 →→→→→→→→→→→→→→→→ 内存地址高")
    print(f"  [{', '.join(map(str, arr.ravel(order='C')))}]")
    print(f"      ↑第0行   ↑第1行   ↑第2行")
    print(f"  同一行的元素在内存中紧挨着: 1旁边是2, 不是5")
    print()

    # F-contiguous 的内存视图
    print(f"  列优先 (F-contiguous) —— 一列一列铺平:")
    print(f"  内存地址低 →→→→→→→→→→→→→→→→ 内存地址高")
    print(f"  [{', '.join(map(str, arr.ravel(order='F')))}]")
    print(f"      ↑第0列  ↑第1列  ↑第2列  ↑第3列")
    print(f"  同一列的元素在内存中紧挨着: 1旁边是5, 不是2")

    print()
    print("  📝 关键认知:")
    print("     CPU 取数据是一次取'一整块'(cache line)")
    print("     如果访问顺序 = 内存排列顺序 → 几乎免费")
    print("     如果访问顺序 ≠ 内存排列顺序 → 每取一个都要跳很远")


def demo_array_layout():
    """第二步：性能对比"""
    print(f"\n{'='*55}")
    print("示例 4b: 内存布局性能对比 (2000×2000)")
    print(f"{'='*55}")

    n, m = 2000, 2000
    arr_c = np.random.randn(n, m).astype(np.float64)     # C-contiguous (默认)
    arr_f = np.asfortranarray(arr_c)                      # F-contiguous

    print(f"  typeof(arr_c): {typeof(arr_c)}")
    print(f"  typeof(arr_f): {typeof(arr_f)}")
    print()

    # 热身
    sum_rows_outer(arr_c[:100, :100])
    sum_cols_outer(arr_c[:100, :100])

    import time

    # C 数组 + 逐行遍历 (内层沿列) → 内存连续访问
    t0 = time.perf_counter()
    r1 = sum_rows_outer(arr_c)
    t1 = time.perf_counter() - t0
    print(f"  ✅ C数组 + 逐行遍历(内层列):  {t1:.4f}s  ← 内存连续，快")

    # C 数组 + 逐列遍历 (内层沿行) → 内存跳跃访问
    t0 = time.perf_counter()
    r2 = sum_cols_outer(arr_c)
    t2 = time.perf_counter() - t0
    print(f"  ❌ C数组 + 逐列遍历(内层行):  {t2:.4f}s  ← 内存跳跃，慢 {t2/t1:.1f}×")

    # F 数组 + 逐列遍历 (内层沿行) → 内存连续访问
    t0 = time.perf_counter()
    r3 = sum_cols_outer(arr_f)
    t3 = time.perf_counter() - t0
    print(f"  ✅ F数组 + 逐列遍历(内层行):  {t3:.4f}s  ← 内存连续，快")

    # F 数组 + 逐行遍历 (内层沿列) → 内存跳跃访问
    t0 = time.perf_counter()
    r4 = sum_rows_outer(arr_f)
    t4 = time.perf_counter() - t0
    print(f"  ❌ F数组 + 逐行遍历(内层列):  {t4:.4f}s  ← 内存跳跃，慢 {t4/t3:.1f}×")

    print()
    print("  📝 黄金法则：让内层循环遍历在内存中紧挨着的那个维度")
    print(f"  C 数组 + 外层列 (cache 不友好):  {t2:.4f}s  ← 慢很多！")

    # 外层列遍历 (对 F 数组友好)
    t0 = time.perf_counter()
    r3 = sum_cols_outer(arr_f)
    t3 = time.perf_counter() - t0
    print(f"  F 数组 + 外层列 (cache 友好):    {t3:.4f}s")


# ──────────────────────────────────────────────
# 示例 5: typed list —— Numba 专用列表
# ──────────────────────────────────────────────

@njit
def typed_list_demo(n):
    """使用 numba.typed.List —— 类型固定、在 njit 内可用"""
    lst = List.empty_list(types.float64)
    for i in range(n):
        lst.append(float(i * i))
    return lst


@njit
def typed_list_init():
    """通过初始值自动推断类型"""
    lst = List([1.0, 2.0, 3.0])
    lst.append(4.0)
    total = 0.0
    for x in lst:
        total += x
    return total


def demo_typed_list():
    print(f"\n{'='*55}")
    print("示例 5: numba.typed.List")
    print(f"{'='*55}")

    result = typed_list_demo(5)
    print(f"  typed_list_demo(5) = {result}")
    print(f"  typeof(result) = {typeof(result)}")

    # 在 Python 端创建 typed list 传给 njit 函数
    py_typed_list = List([10.0, 20.0, 30.0])
    print(f"  外部创建的 typed List: {py_typed_list}")

    print(f"  typed_list_init() = {typed_list_init()}")
    print()
    print("  ⚠️ 普通 Python list 不能直接传给 njit 函数（会触发 reflected list 警告）")
    print("  ✅ 用 numba.typed.List 或 numpy 数组代替")


# ──────────────────────────────────────────────
# 示例 6: typed dict —— Numba 专用字典
# ──────────────────────────────────────────────

@njit
def typed_dict_demo():
    """创建并使用 typed dict"""
    d = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.int64
    )
    d['Alice'] = 85
    d['Bob'] = 92
    d['Charlie'] = 78
    return d


@njit
def dict_lookup(d, key):
    """在 typed dict 中查找"""
    if key in d:                 # ✅ 支持 in 操作
        return d[key]
    return -1


def demo_typed_dict():
    print(f"\n{'='*55}")
    print("示例 6: numba.typed.Dict")
    print(f"{'='*55}")

    d = typed_dict_demo()
    print(f"  typed_dict_demo() = {dict(d)}")   # 转回 Python dict 以便打印
    print(f"  typeof(d) = {typeof(d)}")

    # 在 njit 函数中操作 dict
    d2 = Dict()
    d2['x'] = 10
    d2['y'] = 20
    print(f"  dict_lookup(d2, 'x') = {dict_lookup(d2, 'x')}")
    print(f"  dict_lookup(d2, 'z') = {dict_lookup(d2, 'z')}")

    print()
    print("  📝 typed dict 适合：缓存中间结果、构建查找表")


# ──────────────────────────────────────────────
# 示例 7: 字符串操作
# ──────────────────────────────────────────────

@njit
def string_concat(a, b):
    """字符串拼接"""
    return a + " " + b


@njit
def string_info(s):
    """字符串基本操作"""
    length = len(s)
    first = s[0]
    last = s[-1]
    return length, first, last


def demo_strings():
    print(f"\n{'='*55}")
    print("示例 7: 字符串操作")
    print(f"{'='*55}")

    s = "Numba"
    print(f"  typeof('{s}') = {typeof(s)}")
    print(f"  string_concat('Hello', 'Numba') = '{string_concat('Hello', 'Numba')}'")
    length, first, last = string_info("Hello")
    print(f"  string_info('Hello') = (len={length}, first='{first}', last='{last}')")

    print()
    print("  ✅ 支持: +, len(), s[i], 比较")
    print("  ❌ 不支持: split(), format(), upper(), lower(), strip()")


# ──────────────────────────────────────────────
# 示例 8: 函数类型 —— 高阶函数
# ──────────────────────────────────────────────

@njit
def square(x):
    return x * x


@njit
def cube(x):
    return x * x * x


@njit
def apply_func(arr, func):
    """接受一个函数作为参数"""
    result = np.empty_like(arr)
    for i in range(len(arr)):
        result[i] = func(arr[i])
    return result


def demo_function_types():
    print(f"\n{'='*55}")
    print("示例 8: 函数类型（高阶函数）")
    print(f"{'='*55}")

    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    print(f"  apply_func(arr, square) = {apply_func(arr, square)}")
    print(f"  apply_func(arr, cube)   = {apply_func(arr, cube)}")
    print(f"  typeof(square) = {typeof(square)}")


# ──────────────────────────────────────────────
# 示例 9: inspect_types() —— 深入查看类型推断
# ──────────────────────────────────────────────

@njit
def sample_func(x, arr):
    a = x + 1
    b = a * 2.0
    s = 0.0
    for i in range(len(arr)):
        s += arr[i]
    return b, s


def demo_inspect_types():
    print(f"\n{'='*55}")
    print("示例 9: inspect_types()")
    print(f"{'='*55}")

    # 触发编译
    x = 5
    arr = np.array([1.0, 2.0, 3.0])
    sample_func(x, arr)

    # 查看类型推断的详细结果
    print("  sample_func 的类型推断结果：")
    print("  -" * 25)
    type_info = sample_func.inspect_types()
    # 只打印前几行（避免太长）
    for line in str(type_info).split('\n')[:20]:
        print(f"  {line}")

    # 签名列表
    print(f"\n  已编译的签名: {sample_func.signatures}")


# ──────────────────────────────────────────────
# 示例 10: 可选类型 (Optional)
# ──────────────────────────────────────────────

@njit
def maybe_return(x):
    """返回值可以是 int64 或 None"""
    if x > 0:
        return x * 2
    return None                  # Numba 自动推断返回类型为 Optional(int64)


def demo_optional():
    print(f"\n{'='*55}")
    print("示例 10: 可选类型 (Optional)")
    print(f"{'='*55}")

    print(f"  maybe_return(5)  = {maybe_return(5)}")
    print(f"  maybe_return(-1) = {maybe_return(-1)}")
    print(f"  签名: {maybe_return.signatures}")


# ──────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("╔═══════════════════════════════════════════╗")
    print("║    Numba 第三章：类型系统                 ║")
    print("║    配套代码演示                          ║")
    print("╚═══════════════════════════════════════════╝")

    demo_typeof()
    demo_numeric_types()
    demo_type_unification()

    # 布局演示：先看小矩阵的直观展示，再看性能对比
    demo_memory_layout_visual()
    demo_array_layout()

    demo_typed_list()
    demo_typed_dict()
    demo_strings()
    demo_function_types()
    demo_inspect_types()
    demo_optional()

    print(f"\n{'='*55}")
    print("✅ 第三章代码演示完成！")
    print(f"{'='*55}")
