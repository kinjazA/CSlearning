"""
第四章：支持的 Python 特性 —— 配套代码
==========================================
学习目标：
  1. 确认哪些 Python 语法可以在 @njit 中正常使用
  2. 了解不支持的特性及其替代方案
  3. 学会用 inspect_types() 验证编译结果
"""
import numpy as np
from numba import njit, typeof, types
from numba.typed import List, Dict

# ══════════════════════════════════════════════
# 示例 1: 控制流 —— if / for / while
# ══════════════════════════════════════════════

@njit
def find_threshold(arr, threshold):
    """if/elif/else + for + 提前 return"""
    for i in range(len(arr)):
        val = arr[i]
        if val > threshold:
            return i, 'high'
        elif val < -threshold:
            return i, 'low'
    return -1, 'none'


@njit
def countdown(n):
    """while 循环 + break"""
    result = np.zeros(n, dtype=np.int64)
    i = 0
    while n > 0:
        result[i] = n
        n -= 1
        i += 1
        if n == 5:
            break                    # 提前终止
    return result[:i]


def demo_control_flow():
    print("=" * 55)
    print("示例 1: 控制流")
    print("=" * 55)

    arr = np.array([1.0, -5.0, 3.0, -2.0])
    idx, label = find_threshold(arr, 4.0)
    print(f"  find_threshold → idx={idx}, label='{label}'")

    cd = countdown(10)
    print(f"  countdown(10) → {cd}")


# ══════════════════════════════════════════════
# 示例 2: 内置函数 —— enumerate / zip / abs / round
# ══════════════════════════════════════════════

@njit
def find_first_negative(arr):
    """enumerate 遍历数组"""
    for i, val in enumerate(arr):
        if val < 0:
            return i
    return -1


@njit
def pairwise_sum(a, b):
    """zip 并行遍历两个数组"""
    n = len(a)
    result = np.empty(n, dtype=np.float64)
    for i, (x, y) in enumerate(zip(a, b)):
        result[i] = x + y
    return result


@njit
def builtins_demo(x):
    """abs, min, max, round, bool, int, float"""
    a = abs(x)
    m = min(x, 100.0)
    r = round(x, 2)
    return a, m, r


def demo_builtins():
    print(f"\n{'='*55}")
    print("示例 2: 内置函数")
    print(f"{'='*55}")

    arr = np.array([3.0, -1.0, 5.0, -2.0])
    print(f"  find_first_negative → idx={find_first_negative(arr)}")

    a = np.array([1.0, 2.0, 3.0])
    b = np.array([10.0, 20.0, 30.0])
    print(f"  pairwise_sum → {pairwise_sum(a, b)}")

    print(f"  builtins_demo(-3.14159) → {builtins_demo(-3.14159)}")


# ══════════════════════════════════════════════
# 示例 3: 数据结构 —— 元组、typed list、typed dict
# ══════════════════════════════════════════════

@njit
def tuple_ops(t):
    """元组：索引访问、解包"""
    a, b, c = t                    # 解包
    return a + b + c


@njit
def typed_list_sum(n):
    """在 @njit 内创建并使用 typed list"""
    lst = List.empty_list(types.float64)
    for i in range(n):
        lst.append(float(i * i))
    total = 0.0
    for x in lst:
        total += x
    return total, len(lst)


@njit
def typed_dict_cache(keys, values):
    """用 typed dict 做查找表"""
    d = Dict.empty(
        key_type=types.int64,
        value_type=types.float64
    )
    for i in range(len(keys)):
        d[keys[i]] = values[i]

    # 查找
    result = np.zeros(len(keys), dtype=np.float64)
    for i, k in enumerate(keys):
        result[i] = d[k]           # dict 查询
    return result


def demo_structures():
    print(f"\n{'='*55}")
    print("示例 3: 数据结构")
    print(f"{'='*55}")

    print(f"  tuple_ops((1.0, 2.0, 3.0)) = {tuple_ops((1.0, 2.0, 3.0))}")
    print(f"  typed_list_sum(5) = {typed_list_sum(5)}")

    keys = np.array([10, 20, 30], dtype=np.int64)
    vals = np.array([1.1, 2.2, 3.3], dtype=np.float64)
    print(f"  typed_dict_cache → {typed_dict_cache(keys, vals)}")


# ══════════════════════════════════════════════
# 示例 4: 嵌套函数与闭包
# ══════════════════════════════════════════════

@njit
def outer_function(a, b):
    """嵌套函数"""
    def inner(x):
        return x * x
    return inner(a) + inner(b)


@njit
def make_multiplier(factor):
    """闭包：内部函数捕获外部变量 factor"""
    def multiply(x):
        return x * factor
    return multiply(10)


@njit
def factorial(n):
    """递归"""
    if n <= 1:
        return 1
    return n * factorial(n - 1)


def demo_closures():
    print(f"\n{'='*55}")
    print("示例 4: 嵌套函数 / 闭包 / 递归")
    print(f"{'='*55}")

    print(f"  outer_function(3, 4) = {outer_function(3, 4)}")
    print(f"  make_multiplier(5)  = {make_multiplier(5)}  (10 × 5)")
    print(f"  factorial(6)        = {factorial(6)}")
    print(f"  factorial(20)       = {factorial(20)}")


# ══════════════════════════════════════════════
# 示例 5: 不支持的特性 —— 这些会编译失败
# ══════════════════════════════════════════════

# --- 以下代码不能编译，仅供学习参考 ---

# ❌ try/except
# @njit
# def bad_try_except(a, b):
#     try:
#         return a / b
#     except ZeroDivisionError:
#         return 0

# 替代方案：用 if 提前判断
@njit
def safe_divide(a, b):
    if b == 0:
        return 0.0
    return a / b


# ❌ yield / generator
# @njit
# def bad_generator(n):
#     for i in range(n):
#         yield i * i

# 替代方案：返回数组
@njit
def squares_array(n):
    result = np.empty(n, dtype=np.int64)
    for i in range(n):
        result[i] = i * i
    return result


# ❌ lambda
# @njit
# def bad_lambda(arr):
#     # func = lambda x: x * x     ← 不支持
#     ...

# 替代方案：嵌套函数
@njit
def good_nested_func(arr):
    def square(x):
        return x * x
    result = np.empty_like(arr)
    for i in range(len(arr)):
        result[i] = square(arr[i])
    return result


# ❌ *args / **kwargs
# @njit
# def bad_varargs(*args): ...


# ❌ import 标准库
# @njit
# def bad_import():
#     import json             ← 编译失败


def demo_unsupported():
    print(f"\n{'='*55}")
    print("示例 5: 不支持的特性 & 替代方案")
    print(f"{'='*55}")

    print(f"  safe_divide(10, 0)       = {safe_divide(10.0, 0.0)}  ← if 代替 try/except")
    print(f"  squares_array(5)         = {squares_array(5)}        ← 数组代替 yield")
    print(f"  good_nested_func(arr)     = {good_nested_func(np.array([1.0, 2.0, 3.0]))}  ← 嵌套函数代替 lambda")

    print()
    print("  ❌ try/except    → 用 if/else 提前判断")
    print("  ❌ yield         → 用数组收集结果")
    print("  ❌ lambda        → 用嵌套函数 def")
    print("  ❌ *args/**kwargs → 用固定参数")
    print("  ❌ import 任何库  → 只能 import numpy 和 numba")
    print("  ❌ list/dict/set → 用 numba.typed.List/Dict")


# ══════════════════════════════════════════════
# 示例 6: inspect_types() —— 验证编译结果
# ══════════════════════════════════════════════

@njit
def mixed_operations(x, arr):
    """一个综合函数，检查每个变量的推断类型"""
    a = x + 1
    b = a * 2.0
    s = 0.0
    for i, val in enumerate(arr):
        if val > 0:
            s += val
        else:
            s += abs(val)
    return b, s


def demo_inspect():
    print(f"\n{'='*55}")
    print("示例 6: inspect_types() 验证编译")
    print(f"{'='*55}")

    # 触发编译
    x = 10
    arr = np.array([1.0, -2.0, 3.0])
    mixed_operations(x, arr)

    print(f"  签名: {mixed_operations.signatures}")
    print()

    # 查看类型推断详情（截取关键部分）
    type_str = str(mixed_operations.inspect_types())
    # 只打印变量类型部分
    for line in type_str.split('\n'):
        line = line.strip()
        if '::' in line and not line.startswith('#') and not line.startswith('='):
            print(f"  {line}")

    print()
    print("  💡 养成习惯：写完 @njit 函数就 inspect_types()")
    print("     确认所有变量类型推断正确，没有回退到 object 模式")


# ══════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("╔═══════════════════════════════════════════╗")
    print("║    Numba 第四章：Python 特性支持         ║")
    print("║    配套代码演示                          ║")
    print("╚═══════════════════════════════════════════╝")

    demo_control_flow()
    demo_builtins()
    demo_structures()
    demo_closures()
    demo_unsupported()
    demo_inspect()

    print(f"\n{'='*55}")
    print("✅ 第四章代码演示完成！")
    print(f"{'='*55}")
