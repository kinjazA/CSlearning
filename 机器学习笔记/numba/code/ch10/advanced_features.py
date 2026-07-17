"""
第十章：进阶特性 —— 配套代码
================================
学习目标：
  1. 用 @jitclass 编译 Python 类
  2. 用 @overload 扩展 Numba 函数库
  3. 用 @generated_jit 做类型分发
  4. 对比性能：jitclass vs 普通类
"""
import numpy as np
import time
from numba import njit, jitclass, float64, int64, types
from numba.extending import overload
from numba import generated_jit

# ══════════════════════════════════════════════
# 示例 1: @jitclass 基本用法 —— 编译 Python 类
# ══════════════════════════════════════════════

# 定义字段规范
point_spec = [
    ('x', float64),
    ('y', float64),
]

@jitclass(point_spec)
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        return np.sqrt(dx * dx + dy * dy)

    def move(self, dx, dy):
        self.x += dx
        self.y += dy


def demo_jitclass_basic():
    print("=" * 55)
    print("示例 1: @jitclass 基本用法")
    print("=" * 55)

    p1 = Point(0.0, 0.0)
    p2 = Point(3.0, 4.0)

    print(f"  p1: ({p1.x}, {p1.y})")
    print(f"  p2: ({p2.x}, {p2.y})")
    print(f"  距离: {p1.distance(p2):.2f}")

    p1.move(1.0, 1.0)
    print(f"  p1 移动后: ({p1.x}, {p1.y})")
    print(f"  新距离: {p1.distance(p2):.2f}")

    # 在 @njit 中使用
    @njit
    def compute(p1, p2):
        return p1.distance(p2)

    dist = compute(p1, p2)
    print(f"  @njit 内调用: {dist:.2f}")
    print()
    print("  💡 @jitclass 让类也能编译为机器码")


# ══════════════════════════════════════════════
# 示例 2: @jitclass 数组字段
# ══════════════════════════════════════════════

vector_spec = [
    ('data', float64[:]),
    ('size', int64),
]

@jitclass(vector_spec)
class Vector:
    def __init__(self, n):
        self.data = np.zeros(n, dtype=np.float64)
        self.size = n

    def set(self, i, value):
        if 0 <= i < self.size:
            self.data[i] = value

    def get(self, i):
        if 0 <= i < self.size:
            return self.data[i]
        return 0.0

    def norm(self):
        s = 0.0
        for x in self.data:
            s += x * x
        return np.sqrt(s)


def demo_jitclass_array():
    print(f"\n{'='*55}")
    print("示例 2: @jitclass 数组字段")
    print(f"{'='*55}")

    v = Vector(5)
    for i in range(5):
        v.set(i, float(i + 1))

    print(f"  向量: [{', '.join(str(v.get(i)) for i in range(5))}]")
    print(f"  范数: {v.norm():.4f}")
    print()
    print("  💡 @jitclass 可以有 NumPy 数组字段")


# ══════════════════════════════════════════════
# 示例 3: @jitclass 运算符重载
# ══════════════════════════════════════════════

vec2d_spec = [('x', float64), ('y', float64)]

@jitclass(vec2d_spec)
class Vec2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vec2D(self.x + other.x, self.y + other.y)

    def __mul__(self, scalar):
        return Vec2D(self.x * scalar, self.y * scalar)

    def __str__(self):
        # 注意：__str__ 在 jitclass 中不能直接用，这里仅示意
        return f"Vec2D({self.x}, {self.y})"


def demo_jitclass_operators():
    print(f"\n{'='*55}")
    print("示例 3: @jitclass 运算符重载")
    print(f"{'='*55}")

    v1 = Vec2D(1.0, 2.0)
    v2 = Vec2D(3.0, 4.0)
    v3 = v1 + v2
    v4 = v3 * 2.0

    print(f"  v1: ({v1.x}, {v1.y})")
    print(f"  v2: ({v2.x}, {v2.y})")
    print(f"  v1 + v2: ({v3.x}, {v3.y})")
    print(f"  (v1 + v2) * 2: ({v4.x}, {v4.y})")


# ══════════════════════════════════════════════
# 示例 4: @overload —— 扩展 Numba 函数库
# ══════════════════════════════════════════════

# Python 版本
def my_clamp(x, lo, hi):
    """自定义的 clamp 函数"""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


# Numba 版本
@overload(my_clamp)
def my_clamp_impl(x, lo, hi):
    """为 my_clamp 提供 Numba 实现"""
    if isinstance(x, types.Float):
        def clamp_float(x, lo, hi):
            if x < lo:
                return lo
            if x > hi:
                return hi
            return x
        return clamp_float

    if isinstance(x, types.Integer):
        def clamp_int(x, lo, hi):
            if x < lo:
                return lo
            if x > hi:
                return hi
            return x
        return clamp_int


def demo_overload():
    print(f"\n{'='*55}")
    print("示例 4: @overload 扩展函数")
    print(f"{'='*55}")

    # 在 @njit 中使用自定义函数
    @njit
    def process(arr):
        result = np.empty_like(arr)
        for i in range(len(arr)):
            result[i] = my_clamp(arr[i], 0.0, 1.0)
        return result

    arr = np.array([-0.5, 0.3, 0.7, 1.5, 2.0])
    result = process(arr)

    print(f"  输入: {arr}")
    print(f"  clamp(x, 0, 1): {result}")
    print()
    print("  💡 @overload 让 Numba 认识你自己的函数")


# ══════════════════════════════════════════════
# 示例 5: @generated_jit —— 类型分发
# ══════════════════════════════════════════════

@generated_jit(nopython=True)
def flexible_add(x, y):
    """根据输入类型生成不同实现"""

    if isinstance(x, types.Integer) and isinstance(y, types.Integer):
        # 整数版本
        def add_int(x, y):
            return x + y
        return add_int

    elif isinstance(x, types.Float) or isinstance(y, types.Float):
        # 浮点版本
        def add_float(x, y):
            return x + y
        return add_float


@generated_jit(nopython=True)
def power_optimized(x, n):
    """n 是小常数时展开循环"""
    if isinstance(n, types.Literal):
        n_val = n.literal_value
        if n_val == 2:
            def power_2(x, n):
                return x * x
            return power_2
        elif n_val == 3:
            def power_3(x, n):
                return x * x * x
            return power_3

    # 通用版本
    def power_generic(x, n):
        result = 1.0
        for _ in range(n):
            result *= x
        return result
    return power_generic


def demo_generated_jit():
    print(f"\n{'='*55}")
    print("示例 5: @generated_jit 类型分发")
    print(f"{'='*55}")

    print(f"  flexible_add(1, 2) = {flexible_add(1, 2)}  (int)")
    print(f"  flexible_add(1.0, 2.0) = {flexible_add(1.0, 2.0)}  (float)")

    print(f"  power_optimized(3.0, 2) = {power_optimized(3.0, 2)}")
    print(f"  power_optimized(3.0, 3) = {power_optimized(3.0, 3)}")
    print()
    print("  💡 @generated_jit 根据类型选择不同实现")


# ══════════════════════════════════════════════
# 示例 6: 性能对比 —— @jitclass vs 普通类
# ══════════════════════════════════════════════

# 普通 Python 类
class PointPython:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        return np.sqrt(dx * dx + dy * dy)


def demo_performance():
    print(f"\n{'='*55}")
    print("示例 6: @jitclass vs 普通类性能对比")
    print(f"{'='*55}")

    n = 100000

    # Python 类
    points_py = [PointPython(float(i), float(i+1)) for i in range(n)]
    t0 = time.perf_counter()
    total_py = sum(points_py[i].distance(points_py[i+1]) for i in range(n-1))
    t_py = time.perf_counter() - t0

    # Jitclass（在 @njit 内使用）
    @njit
    def compute_jit(points):
        total = 0.0
        for i in range(len(points) - 1):
            total += points[i].distance(points[i+1])
        return total

    points_jit = [Point(float(i), float(i+1)) for i in range(n)]
    compute_jit(points_jit[:10])  # 热身

    t0 = time.perf_counter()
    total_jit = compute_jit(points_jit)
    t_jit = time.perf_counter() - t0

    print(f"  计算次数: {n-1:,}")
    print(f"  Python 类: {t_py:.4f}s")
    print(f"  @jitclass: {t_jit:.4f}s  ← 快 {t_py/t_jit:.0f}×")
    print(f"  结果一致: {abs(total_py - total_jit) < 1e-6}")


# ══════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("╔═══════════════════════════════════════════╗")
    print("║    Numba 第十章：进阶特性                 ║")
    print("║    配套代码演示                          ║")
    print("╚═══════════════════════════════════════════╝")

    demo_jitclass_basic()
    demo_jitclass_array()
    demo_jitclass_operators()
    demo_overload()
    demo_generated_jit()
    demo_performance()

    print(f"\n{'='*55}")
    print("✅ 第十章代码演示完成！")
    print(f"{'='*55}")
