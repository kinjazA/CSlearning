"""
第十一章：调试与诊断 —— 配套代码
=====================================
学习目标：
  1. 用 inspect_types() 排查类型推断问题
  2. 识别常见编译错误
  3. 定位性能瓶颈
  4. 学会系统化的调试流程
"""
import numpy as np
import time
from numba import njit, types
from numba.typed import List

# ══════════════════════════════════════════════
# 示例 1: TypingError —— 类型推断失败
# ══════════════════════════════════════════════

def demo_typing_error():
    print("=" * 55)
    print("示例 1: TypingError —— 类型推断失败")
    print("=" * 55)

    # ❌ 错误示例（会失败）
    print("  ❌ 错误代码:")
    print("     @njit")
    print("     def bad_typing(arr):")
    print("         result = []              # Python list")
    print("         for x in arr:")
    print("             result.append(x)")
    print("         return result")
    print()
    print("  错误: TypingError - Python list 不支持")

    # ✅ 修复版本
    @njit
    def good_typing(arr):
        result = np.empty_like(arr)
        for i in range(len(arr)):
            result[i] = arr[i] * 2
        return result

    arr = np.array([1.0, 2.0, 3.0])
    print(f"\n  ✅ 修复: 用 NumPy 数组代替 Python list")
    print(f"     result = {good_typing(arr)}")


# ══════════════════════════════════════════════
# 示例 2: inspect_types() —— 查看类型推断
# ══════════════════════════════════════════════

@njit
def example_function(a, b):
    x = a + b
    y = x * 2
    z = y ** 2
    return z


def demo_inspect_types():
    print(f"\n{'='*55}")
    print("示例 2: inspect_types() 查看类型推断")
    print(f"{'='*55}")

    # 触发编译
    result = example_function(1, 2)
    print(f"  example_function(1, 2) = {result}")

    print(f"\n  函数签名: {example_function.signatures}")

    # 获取类型推断信息
    type_info = str(example_function.inspect_types())

    # 提取关键行
    print(f"\n  类型推断关键信息:")
    for line in type_info.split('\n'):
        line = line.strip()
        if '::' in line and not line.startswith('#'):
            # 简化输出，只显示变量类型
            if 'arg(' in line or '=' in line:
                print(f"    {line[:60]}")

    print()
    print("  💡 inspect_types() 用于排查类型推断问题")
    print("     搜索 'pyobject' → 找到回退到 object 模式的地方")


# ══════════════════════════════════════════════
# 示例 3: 定位 object 模式回退
# ══════════════════════════════════════════════

@njit
def slow_with_list(arr):
    """故意用 Python list 导致性能下降"""
    result = List.empty_list(types.float64)
    for x in arr:
        result.append(x * 2)
    return result


@njit
def fast_with_array(arr):
    """用 NumPy 数组"""
    result = np.empty_like(arr)
    for i in range(len(arr)):
        result[i] = arr[i] * 2
    return result


def demo_object_mode_detection():
    print(f"\n{'='*55}")
    print("示例 3: 定位 object 模式回退")
    print(f"{'='*55}")

    arr = np.random.randn(100000).astype(np.float64)

    # 热身
    slow_with_list(arr[:10])
    fast_with_array(arr[:10])

    t0 = time.perf_counter()
    r1 = slow_with_list(arr)
    t1 = time.perf_counter() - t0

    t0 = time.perf_counter()
    r2 = fast_with_array(arr)
    t2 = time.perf_counter() - t0

    print(f"  typed.List 版本: {t1:.4f}s")
    print(f"  NumPy 数组版本:  {t2:.4f}s  ← 快 {t1/t2:.1f}×")

    print()
    print("  💡 用 inspect_types() 排查:")
    print("     如果看到变量是 'List[float64]' → 没问题")
    print("     如果看到 'pyobject' → 回退，性能差")


# ══════════════════════════════════════════════
# 示例 4: 常见性能陷阱 —— 循环内分配
# ══════════════════════════════════════════════

@njit
def bad_alloc_loop(n):
    """❌ 循环内分配内存"""
    result = 0.0
    for i in range(n):
        temp = np.zeros(100)           # 每次分配 → 慢
        result += np.sum(temp)
    return result


@njit
def good_alloc_once(n):
    """✅ 循环外分配一次"""
    temp = np.zeros(100)
    result = 0.0
    for i in range(n):
        result += np.sum(temp)
    return result


def demo_alloc_trap():
    print(f"\n{'='*55}")
    print("示例 4: 性能陷阱 —— 循环内分配")
    print(f"{'='*55}")

    n = 10000

    # 热身
    bad_alloc_loop(10)
    good_alloc_once(10)

    t0 = time.perf_counter()
    r1 = bad_alloc_loop(n)
    t1 = time.perf_counter() - t0

    t0 = time.perf_counter()
    r2 = good_alloc_once(n)
    t2 = time.perf_counter() - t0

    print(f"  ❌ 循环内分配: {t1:.4f}s")
    print(f"  ✅ 循环外分配: {t2:.4f}s  ← 快 {t1/t2:.1f}×")
    print()
    print("  💡 内存分配比计算贵 100-1000 倍")


# ══════════════════════════════════════════════
# 示例 5: 类型不稳定
# ══════════════════════════════════════════════

@njit
def unstable_type(n):
    """❌ 类型在循环中变化"""
    x = 0
    for i in range(n):
        if i % 2 == 0:
            x = i                      # int
        else:
            x = float(i)               # float ← 类型切换
    return x


@njit
def stable_type(n):
    """✅ 类型统一"""
    x = 0.0                            # 统一用 float
    for i in range(n):
        x = float(i)
    return x


def demo_type_stability():
    print(f"\n{'='*55}")
    print("示例 5: 类型稳定性")
    print(f"{'='*55}")

    n = 1000000

    # 热身
    unstable_type(10)
    stable_type(10)

    t0 = time.perf_counter()
    r1 = unstable_type(n)
    t1 = time.perf_counter() - t0

    t0 = time.perf_counter()
    r2 = stable_type(n)
    t2 = time.perf_counter() - t0

    print(f"  ❌ 类型不稳定: {t1:.4f}s")
    print(f"  ✅ 类型统一:   {t2:.4f}s  ← 快 {t1/t2:.1f}×")
    print()
    print("  💡 用 inspect_types() 检查变量类型是否一致")


# ══════════════════════════════════════════════
# 示例 6: 调试流程演示
# ══════════════════════════════════════════════

def demo_debug_workflow():
    print(f"\n{'='*55}")
    print("示例 6: 调试流程")
    print(f"{'='*55}")

    print("""
  Numba 函数不工作？按此流程排查：

  1️⃣ 能编译吗？
     - 看错误类型（TypingError / LoweringError）
     - 检查是否用了不支持的特性

  2️⃣ 能编译但很慢？
     - inspect_types() 搜索 'pyobject'
     - 检查循环内是否有分配
     - 检查类型是否稳定

  3️⃣ 结果错误？
     - 设置 NUMBA_DISABLE_JIT=1
     - 用 Python 调试器（pdb / IDE）
     - 对比 Numba 版本和纯 Python 版本

  4️⃣ 性能不如预期？
     - 用 line_profiler 找热点行
     - 检查是否能向量化
     - 考虑 parallel=True
    """)


# ══════════════════════════════════════════════
# 示例 7: 环境变量提示
# ══════════════════════════════════════════════

def demo_environment_vars():
    print(f"\n{'='*55}")
    print("示例 7: 调试用环境变量")
    print(f"{'='*55}")

    print("""
  调试时可用的环境变量：

  🔧 NUMBA_DISABLE_JIT=1
     禁用编译，直接用 Python 解释器
     用途：用 pdb/IDE 调试逻辑错误

  🔧 NUMBA_WARNINGS=1
     显示所有警告
     用途：排查隐藏的性能问题

  🔧 NUMBA_DUMP_ANNOTATION=1
     导出类型注解到 HTML
     用途：可视化查看类型推断

  用法 (Linux/Mac):
    export NUMBA_DISABLE_JIT=1
    python your_script.py

  用法 (Windows):
    set NUMBA_DISABLE_JIT=1
    python your_script.py
    """)


# ══════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("╔═══════════════════════════════════════════╗")
    print("║    Numba 第十一章：调试与诊断             ║")
    print("║    配套代码演示                          ║")
    print("╚═══════════════════════════════════════════╝")

    demo_typing_error()
    demo_inspect_types()
    demo_object_mode_detection()
    demo_alloc_trap()
    demo_type_stability()
    demo_debug_workflow()
    demo_environment_vars()

    print(f"\n{'='*55}")
    print("✅ 第十一章代码演示完成！")
    print(f"{'='*55}")
