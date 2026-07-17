# 第二章：基础用法——JIT 装饰器

> **核心问题**：`@njit` 一行代码就完成加速，但它背后有哪些控制选项？什么时候会加速失败？如何精准掌控编译行为？

---

## 2.1 `@jit` 装饰器入门

### 2.1.1 基本语法

```python
from numba import jit

@jit                          # 最简形式
def func1(x, y):
    return x + y

@jit()                        # 等价于上面，但可以传入参数
def func2(x, y):
    return x + y

@jit(nopython=True)           # 强制 nopython 模式
def func3(x, y):
    return x + y
```

### 2.1.2 第一次调用：热身（Warm-up）

> 如果你还没理解"热身"是什么，请先阅读第 1 章 §1.4——那里有一个寿司店的类比能帮你建立直观感受。

```python
@njit
def my_func(arr):
    total = 0
    for x in arr:
        total += x
    return total

# 此时 my_func 还没被编译，Python 层存的是一个"编译分发器"的引用

result = my_func(np.array([1, 2, 3]))  # ← 这一行触发热身（编译 + 执行）
# 1. Numba 看到输入类型: Array(int64, 1, C)
# 2. 类型推断引擎遍历函数 IR，推断所有变量类型
# 3. 生成 LLVM IR → 优化 → 机器码
# 4. 存入内存缓存
# 5. 执行机器码，返回结果
```

关键认知：
- **编译发生在运行时**（JIT = Just-In-Time），不是在 `import` 时
- **编译成本是一次性的**，按 (函数, 输入类型组合) 缓存。同类型第二次调用 → 瞬间完成
- 对短小的、只跑一次的函数，热身时间可能超过运行时间——不值得
- 可以用 `cache=True` 把编译结果存到磁盘，下次启动时跳过热身

### 2.1.3 适合 Numba 加速的函数特征

| 特征 | 适合 | 不适合 |
|------|------|--------|
| 调用次数 | 多次调用（编译成本摊薄） | 只调用一次 |
| 代码结构 | 循环密集、数值计算 | I/O、字符串、网络 |
| 数据类型 | NumPy 数组、数值标量 | Python 对象、自定义类 |
| 复杂度 | 几十行到几百行 | 一行 `np.sum()` |

---

## 2.2 nopython 模式 vs object 模式

这是 Numba 中最容易混淆的概念，需要彻底搞懂。

### 2.2.1 nopython 模式（推荐）

```python
@jit(nopython=True)   # 或简写 @njit
def func():
    ...
```

**行为**：
- Numba **必须**把函数中所有语句都编译为机器码
- 如果遇到不支持的操作 → **立即抛出异常**（TypingError、LoweringError）
- **不经过** Python C API → 极快

**编译成功 = 跑得快；编译失败 = 立即报错**

### 2.2.2 object 模式（备选/不推荐）

```python
@jit                  # 默认 nopython=False，允许回退
def func():
    ...
```

**行为**：
- Numba 尽力编译，编译不了的 → 回退为 Python C API 调用
- 不会立即报错，但会发出 `NumbaPerformanceWarning`
- 回退的部分**比纯 Python 还慢**（因为多了跨编译界面的切换开销）

```python
@jit
def mixed_func(arr):
    total = 0
    for x in arr:          # ✅ 能 nopython 编译
        total += x
    result = some_pd_func()  # ❌ pandas 不能编译 → 回退到 object 模式
    return total, result
```

**结论：`@jit` 不带 `nopython=True` 是自欺欺人——你以为在加速，实际可能更慢。永远用 `@njit`。**

### 2.2.3 强制装饰器行为对比

```python
from numba import njit, jit

@njit                         # 等价于 @jit(nopython=True)
def strict_func(x): ...

@jit(nopython=True)           # 同上，显式写法
def strict_func2(x): ...

@jit                          # ⚠️ 允许回退到 object 模式（不推荐！）
def loose_func(x): ...

@jit(forceobj=True)           # 强制 object 模式（纯 Python 速度）
def objonly_func(x): ...
```

---

## 2.3 `@njit` —— 你应该默认使用的装饰器

```python
from numba import njit

@njit
def my_function(arr):
    ...
```

`@njit` = `@jit(nopython=True)`。**全 Numba 社区共识：除非你有明确的理由用 object 模式，否则一律 `@njit`。**

---

## 2.4 函数签名（Signature）

### 2.4.1 为什么需要显式签名

Numba 默认依赖**类型推断**——在第一次调用时根据实际参数类型推导所有变量类型。这有代价：

1. **首次调用需要编译** → 有延迟
2. **类型推断可能不符合预期** → 比如你希望 `float32` 但推断为 `float64`
3. **延迟到运行时才暴露类型错误** → 测试才能发现

显式签名可以在**函数定义时**就确定类型，不需要等第一次调用来推断。

### 2.4.2 签名字符串语法

```python
from numba import njit, int64, float64, boolean

# 完整语法：返回类型(参数类型1, 参数类型2, ...)
@njit(float64(float64, float64))
def add(a, b):
    return a + b

# 无返回值
@njit(boolean(int64[:]))      # int64[:] 是一维 int64 数组
def has_negative(arr):
    for x in arr:
        if x < 0:
            return True
    return False

# 返回数组
@njit(float64[:](float64[:], float64[:]))
def add_arrays(a, b):
    return a + b
```

### 2.4.3 内置类型缩写一览

| 缩写 | 含义 |
|------|------|
| `b1` | `boolean` (8-bit) |
| `i1` / `i2` / `i4` / `i8` | `int8` / `int16` / `int32` / `int64` |
| `u1` / `u2` / `u4` / `u8` | `uint8` / `uint16` / `uint32` / `uint64` |
| `f4` / `f8` | `float32` / `float64` |
| `c8` / `c16` | `complex64` / `complex128` |
| `void` | 无返回值 |

### 2.4.4 数组签名语法

```python
# 一维数组: type[:]
@njit(float64(float64[:]))
def sum1d(arr): ...

# 二维数组: type[:, :]
@njit(float64(float64[:, :]))
def sum2d(arr): ...

# 三维数组: type[:, :, :]
@njit(float64(float64[:, :, :]))
def sum3d(arr): ...

# 多维 + 非连续: type[::1]  表示 C-contiguous (行优先)
@njit(float64(float64[::1]))
def sum_c_contiguous(arr): ...

# F-contiguous (列优先)
@njit(float64(float64[:, ::1]))
def col_sum(arr): ...
```

### 2.4.5 多签名（一个函数，多种类型）

```python
from numba import njit, float32, float64

# 方式1：列出所有签名
@njit([float32(float32, float32),
       float64(float64, float64)])
def multiply(a, b):
    return a * b

# 方式2：generated_jit —— 根据输入类型"生成"代码（高级，第10章详讲）
```

---

## 2.5 编译选项详解

`@njit` 除了基础用法外，还有一系列控制编译行为的选项：

### 2.5.1 `cache=True` — 编译缓存

```python
@njit(cache=True)
def expensive_loop(arr):
    ...
```

| 维度 | 说明 |
|------|------|
| **行为** | 首次编译后将结果写入 `__pycache__/` 目录下的 `.nbc` 文件 |
| **效果** | 下次启动程序时直接加载磁盘缓存，跳过编译 |
| **代价** | 略慢的首次编译（多一次磁盘写入）；占用少量磁盘空间 |
| **适用** | 函数相对稳定、被多个脚本/多次调用 |
| **不适用** | 频繁修改的函数（缓存失效频繁）、一次性脚本 |
| **清理** | 删除 `__pycache__/` 目录或 `.nbc` 文件即可 |

> ⚠️ 如果函数依赖的闭包变量变了，缓存会自动失效（Numba 用源码哈希检测变化）。

### 2.5.2 `inline='always'` — 函数内联

```python
@njit(inline='always')
def helper(a):
    return a * a + 1

@njit
def main_func(arr):
    result = 0
    for x in arr:
        result += helper(x)    # helper 的代码被"嵌入"到这里
    return result
```

内联 = 消除函数调用的开销（参数传递、栈帧切换）。适用场景：
- 被频繁调用的小函数
- 循环内调用的辅助函数
- 代价：代码体积膨胀（每个调用点都复制一份）

```python
# inline 选项：
@njit(inline='always')    # 强制内联
@njit(inline='never')     # 禁止内联（默认）
```

### 2.5.3 `fastmath=True` — 快数模式

```python
@njit(fastmath=True)
def compute(arr):
    return np.sqrt(np.sum(arr ** 2))
```

`fastmath` 会放松 IEEE 754 浮点标准的保证来换取速度：

| 放宽的保证 | 影响 |
|-----------|------|
| 不使用 `NaN` / `Inf` 检查 | 浮点异常不触发，直接产生错误结果 |
| 允许对运算重排序 | `(a+b)+c` ≠ `a+(b+c)` 可能发生 |
| 不保证 `-0.0` 和 `+0.0` 区别 | 这两者被视为相等 |
| 允许使用 reciprocal 近似 | `x/y` → `x*(1/y)`，精度略降 |
| 允许使用 FMA (fused multiply-add) | 精度更高但结果与无 FMA 时不同 |

### 2.5.4 `error_model` — 浮点异常处理策略

```python
@njit(error_model='numpy')    # 默认：除零 → NaN/Inf（和 NumPy 一致）
def numpy_style(arr):
    return 1.0 / arr

@njit(error_model='python')   # 除零 → 抛出 ZeroDivisionError（和 Python 一致）
def python_style(arr):
    return 1.0 / arr
```

| 值 | 行为 |
|----|------|
| `'numpy'`（默认） | 除零产生 `NaN`/`Inf`，不抛异常 |
| `'python'` | 除零抛出 `ZeroDivisionError` |

### 2.5.5 `boundscheck=False` — 数组越界检查

```python
@njit(boundscheck=True)    # 开发阶段：检查越界
def dev_func(arr):
    return arr[100]         # 越界 → 抛出 IndexError

@njit(boundscheck=False)   # 生产阶段：不检查越界（更快，但越界时行为未定义）
def prod_func(arr):
    return arr[100]         # 越界 → 可能 segfault！
```

默认行为：在 `@njit` 下，如果数组索引**可以静态证明**不越界，则省略检查；否则保留检查。

### 2.5.6 `parallel=True` — 自动并行化

```python
@njit(parallel=True)
def parallel_sum(arr):
    n = len(arr)
    total = 0
    for i in prange(n):     # 注意：需要用 prange 而不是 range
        total += arr[i]
    return total
```

⚠️ 自动并行化是高级主题，**第八章**会详细讲 `prange` 和并行化的注意事项。并非所有循环都能安全地并行化。

### 2.5.7 `nogil=True` — 释放 GIL

```python
@njit(nogil=True)
def cpu_intensive(arr):
    # 纯 Numba 代码，不涉及 Python 对象 → 运行时不持有 GIL
    ...
```

效果：这个函数执行时**不持有 GIL** →可以在多线程中真正做到并行。这对 `concurrent.futures.ThreadPoolExecutor` 等场景非常有用。

---

## 2.6 编译过程的"内省"

Numba 提供了查看编译结果的工具：

```python
from numba import njit

@njit
def add(x, y):
    return x + y

# 触发编译
add(1, 2)

# 查看编译生成的 LLVM IR
print(add.inspect_llvm())

# 查看编译生成的汇编代码
print(add.inspect_asm())

# 查看类型推断结果
print(add.inspect_types())

# 查看所有已编译的签名
print(add.signatures)   # [(int64, int64)]
```

---

## 2.7 常见问题与检查清单

| 问题 | 原因 | 解决 |
|:----:|:----:|:----:|
| 首次调用慢 | 编译开销 | `cache=True`；或预先用 dummy 数据"预热" |
| `@njit` 报 TypingError | 用了 Numba 不支持的操作 | 查官方文档支持列表；拆分函数 |
| `@jit` 下 NumbaPerformanceWarning | 部分代码回退到 object 模式 | 改用 `@njit` 报错定位问题 |
| 编译后没变快 | 函数中没有可加速的循环/数值计算 | 检查代码；不是所有代码都适合 JIT |
| 数组签名不匹配（慢很多） | 布局不匹配导致拷贝 | 用 `np.ascontiguousarray` 确保 C-contiguous |

---

## 2.8 本章关键概念

| 概念 | 一句话 |
|:----:|:------:|
| `@njit` | `@jit(nopython=True)`，应该始终使用的默认装饰器 |
| nopython 模式 | 全部编译为机器码，失败就报错——但不自欺欺人 |
| object 模式 | 编译不了的退回 Python API——通常更慢，不推荐 |
| 函数签名 | 显式声明输入类型和返回类型，跳过类型推断 |
| 编译缓存 | 缓存到磁盘，跨会话生效 |
| fastmath | 牺牲浮点精度换速度，科学计算中要谨慎 |
| 首次调用开销 | 编译的时间代价——长函数可超 1 秒，需要预热策略 |

---

## 2.9 下章预告

第三章将深入 Numba 的类型系统：数值类型、数组类型、复合类型、类型推断的边界，以及如何用 `numba.typeof()` 排查类型问题。
