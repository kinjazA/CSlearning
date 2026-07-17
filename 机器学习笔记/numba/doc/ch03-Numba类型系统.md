# 第三章：Numba 类型系统

> **核心问题**：Numba 如何"看懂"你的变量是什么类型？类型推断在什么情况下会失败？如何用 `numba.typeof()` 排查问题？

---

## 3.1 为什么类型对 Numba 如此重要

### 3.1.1 JIT 编译的本质

Numba 要生成机器码，而机器码要求每个变量有**确定的内存布局**——占多少字节、怎么解释这些字节。Python 的 `int` 可以无限大（大整数对象），C 的 `int64_t` 永远是 8 字节。JIT 编译器必须在**编译时**知道每个变量的精确类型，否则无法生成高效机器码。

```
Python 运行时：
  a = 1     # PyLongObject * → 可以变成 a = "hello"

Numba 编译时：
  a = 1     # 必须确定是 int32? int64? float64?
             # 编译完后就不能变了
```

### 3.1.2 类型推断三部曲

```
输入参数类型（从实际传入值获取）
        │
        ▼
  ┌─────────────────┐
  │ 向前传播类型     │  a: int64, b: int64
  │ (Forward Prop)  │  c = a + b → c: int64
  └─────────────────┘
        │
        ▼
  ┌─────────────────┐
  │ 向后约束类型     │  if isinstance(x, float): ...
  │ (Backward Prop) │  → x 被约束为 float 类型
  └─────────────────┘
        │
        ▼
  ┌─────────────────┐
  │ 统一与求解       │  如果推断不一致 → TypingError
  │ (Unification)   │  如果一致 → 生成 IR
  └─────────────────┘
```

### 3.1.3 类型推断的典型失败场景

```python
@njit
def ambiguous(a):
    if a > 0:
        result = 1        # int64
    else:
        result = 2.0      # float64
    return result          # ❌ result 是 int64 还是 float64？→ TypingError
```

**正确做法：在分支前确定类型**

```python
@njit
def fixed(a):
    result = 0.0          # 先声明为 float64
    if a > 0:
        result = 1.0      # float64
    else:
        result = 2.0      # float64
    return result          # ✅ 统一了
```

---

## 3.2 数值类型

### 3.2.1 整数类型

| Numba 类型 | Python 对应 | 字节数 | 值范围 |
|:---------:|:----------:|:------:|:------:|
| `int8` / `i1` | — | 1 | -128 ~ 127 |
| `int16` / `i2` | — | 2 | -32,768 ~ 32,767 |
| `int32` / `i4` | — | 4 | -2³¹ ~ 2³¹-1 |
| `int64` / `i8` | `int` | 8 | -2⁶³ ~ 2⁶³-1 |
| `uint8` / `u1` | — | 1 | 0 ~ 255 |
| `uint16` / `u2` | — | 2 | 0 ~ 65,535 |
| `uint32` / `u4` | — | 4 | 0 ~ 2³²-1 |
| `uint64` / `u8` | — | 8 | 0 ~ 2⁶⁴-1 |

**默认行为**：Python 的 `int` 在 Numba 中默认推断为 `int64`（64 位系统）。

**整数溢出**：Numba 中的 `int64` 有溢出行为（和 C 一样），与 Python 的任意精度 `int` 不同：

```python
import numpy as np

@njit
def overflow_demo():
    x = np.int64(9223372036854775807)  # int64 最大值
    return x + 1                        # 溢出回绕为最小值！
```

### 3.2.2 浮点类型

| Numba 类型 | Python 对应 | 字节数 | 精度 |
|-----------|------------|--------|------|
| `float32` / `f4` | — | 4 | ~7 位有效数字 |
| `float64` / `f8` | `float` | 8 | ~15 位有效数字 |

- Python 的 `float` 默认推断为 `float64`
- 如果 NumPy 数组是 `float32`，Numba 会保持 `float32`

### 3.2.3 复数类型

| Numba 类型 | 字节数 |
|-----------|--------|
| `complex64` / `c8` | 8 (两个 float32) |
| `complex128` / `c16` | 16 (两个 float64) |

### 3.2.4 布尔类型

- `boolean`（别名 `b1`）：1 字节，值为 `True` / `False`
- 注意：Numba 中 `bool` 是 1 字节，NumPy 的 `np.bool_` 也是 1 字节

---

## 3.3 数组类型

这是 Numba 中最重要的类型——几乎所有科学计算都围绕数组展开。

### 3.3.1 数组类型的三要素

```
Array(dtype, ndim, layout)
       ↑       ↑      ↑
    元素类型  维度数  内存布局
```

一个 `np.array([[1.0, 2.0], [3.0, 4.0]])` 在 Numba 中表示为：

```
Array(float64, 2, 'C')
   ↑          ↑   ↑
 float64    二维  行优先(C-contiguous)
```

### 3.3.2 数组签名语法（回顾+深化）

```python
float64[:]          # 一维 float64 数组
float64[:, :]       # 二维 float64 数组
int64[:, :, :]      # 三维 int64 数组
float32[:]          # 一维 float32 数组
```

**内存布局修饰符**：

```python
float64[::1]              # 一维 C-contiguous (行优先)
float64[:, ::1]           # 二维：第二维是 C-contiguous
float64[::1, :]           # 二维：第一维是 F-contiguous (列优先)
```

### 3.3.3 行优先 (C-contiguous) vs 列优先 (F-contiguous)

> 这是 Numba 性能优化的关键概念，如果不理解，你的 Numba 代码可能反而比 NumPy 慢。

#### 第一步：内存不是二维的

你在 Python 里创建了一个 `3×4` 的矩阵：

```python
arr = np.array([[1,  2,  3,  4],
                [5,  6,  7,  8],
                [9, 10, 11, 12]])
```

这只是你看的样子。**计算机内存里没有"行"和"列"**，只有一根长长的、一维的地址线。上面的 12 个数字必须一个接一个地排进这根线里。

问题是：**按什么顺序排？**

#### 第二步：两种"铺平"方式

**行优先（C-contiguous）—— 一行一行地铺**

这是 NumPy **默认**的方式，因为 Python 底层是 C 语言。

把矩阵像写文章一样，一行一行往下排：

```
矩阵：
┌            ┐
│  1  2  3  4│   ← 第 0 行
│  5  6  7  8│   ← 第 1 行
│  9 10 11 12│   ← 第 2 行
└            ┘

内存地址增大的方向 →→→→→→→→→→→→→→→→→→→→→→→
┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │10 │11 │12 │
└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
  ↑第0行开始    ↑第1行开始      ↑第2行开始
```

**关键规律**：同一行里的元素，在内存里**紧挨着**。`1` 的下一个是 `2`（同一行的下一列），不是 `5`（下一行）。

**列优先（F-contiguous）—— 一列一列地铺**

Fortran / MATLAB 使用的方式。把矩阵竖着往下排：

```
矩阵：
┌            ┐
│  1  2  3  4│
│  5  6  7  8│
│  9 10 11 12│
└            ┘

按列铺平：
内存地址增大的方向 →→→→→→→→→→→→→→→→→→→→→→→
┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│ 1 │ 5 │ 9 │ 2 │ 6 │10 │ 3 │ 7 │11 │ 4 │ 8 │12 │
└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
  ↑第0列    ↑第1列    ↑第2列    ↑第3列
```

**关键规律**：同一列里的元素，在内存里**紧挨着**。`1` 的下一个是 `5`（同一列的下一行），不是 `2`（下一列）。

#### 第三步：这对性能意味着什么？

CPU 从内存取数据不是一个个取的，而是一次取**一整块**（叫 cache line，通常 64 字节）。这一块里相邻的数据会一起进入 CPU 的高速缓存。取一次很贵，但取相邻的几乎免费。

**场景：遍历一个 `3×4` 的 C-contiguous 数组**

```
内存布局：[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
              ↑─ cache line ─↑  ↑─ 下一次取 ─↑

arr[0] = [1, 2, 3, 4]     ← 这四个数字在内存里紧挨着，一次 cache line 全拿到
arr[1] = [5, 6, 7, 8]     ← 这四个也紧挨着，又一次全拿到
arr[2] = [9, 10, 11, 12]  ← 同上
```

| 循环写法 | 访问顺序 | 内存中的实际跳转 | 缓存命中 |
|---------|---------|---------------|---------|
| **逐行遍历** `arr[i][j]` | 1→2→3→4→5→6→... | 紧挨着走，不跳 | ✅ 几乎每次都命中 |
| **逐列遍历** `arr[j][i]` | 1→5→9→2→6→10→... | 每次跳跃 4 个位置 | ❌ 频繁 cache miss |

**逐列遍历一个 C-contiguous 数组就像**：你在一本书里找字，但不是按顺序读——而是先读每行的第一个字，读完了再回头读每行的第二个字……每次都要翻页。CPU 也一样辛苦。

#### 第四步：一条黄金规则

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   让"内层循环"遍历"在内存里紧挨着"的那个维度             │
│                                                         │
│   C-contiguous：最后一维相邻 → 内层循环遍历最后一维       │
│   F-contiguous：第一维相邻 → 内层循环遍历第一维           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

具体来说：

```python
# ✅ C-contiguous 数组的正确遍历方式：外层行，内层列
#    因为 arr[i, j] 和 arr[i, j+1] 在内存中紧挨着
for i in range(n):          # 外层：行
    for j in range(m):      # 内层：列 ← 在内存中连续访问
        total += arr[i, j]

# ❌ 错误遍历方式：外层列，内层行
#    每次内层循环 arr[j, i] 跳到 arr[j+1, i]，在内存中隔了一整行
for j in range(m):          # 外层：列
    for i in range(n):      # 内层：行 ← 在内存中跳跃访问
        total += arr[j, i]
```

#### 第五步：用 Numba 验证

```python
import numpy as np
import time
from numba import njit, typeof

@njit
def sum_row_first(arr):      # 外层行、内层列 → C 数组友好
    s = 0.0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            s += arr[i, j]
    return s

@njit
def sum_col_first(arr):      # 外层列、内层行 → F 数组友好
    s = 0.0
    for j in range(arr.shape[1]):
        for i in range(arr.shape[0]):
            s += arr[i, j]
    return s

arr_c = np.random.randn(5000, 5000)           # C-contiguous (默认)
arr_f = np.asfortranarray(arr_c)              # F-contiguous（列优先）

# 热身
sum_row_first(arr_c[:10, :10])
sum_col_first(arr_c[:10, :10])

# C 数组 + 逐行遍历（正确）→ 快
t0 = time.perf_counter()
sum_row_first(arr_c)
print(f"C数组+逐行: {time.perf_counter()-t0:.3f}s")   # 快

# C 数组 + 逐列遍历（错误）→ 慢 3-5 倍
t0 = time.perf_counter()
sum_col_first(arr_c)
print(f"C数组+逐列: {time.perf_counter()-t0:.3f}s")   # 慢！

# F 数组 + 逐列遍历（正确）→ 快
t0 = time.perf_counter()
sum_col_first(arr_f)
print(f"F数组+逐列: {time.perf_counter()-t0:.3f}s")   # 快

# F 数组 + 逐行遍历（错误）→ 慢
t0 = time.perf_counter()
sum_row_first(arr_f)
print(f"F数组+逐行: {time.perf_counter()-t0:.3f}s")   # 慢！
```

#### 小结

| | 行优先 (C-contiguous) | 列优先 (F-contiguous) |
|--|---------------------|---------------------|
| **别名** | C 风格、行主序 | Fortran 风格、列主序 |
| **NumPy 默认** | ✅ 是 | ❌ 需要 `order='F'` |
| **内存中紧挨着** | 同一**行**的元素 | 同一**列**的元素 |
| **最快遍历方向** | 内层循环遍历**列**（最后一个维度） | 内层循环遍历**行**（第一个维度） |
| **典型来源** | 直接从 Python 创建、np.zeros、np.ones | `np.asfortranarray()`、从 MATLAB 读取 |
| **Numba 签名** | `float64[::1]` 或 `float64[:, ::1]` | `float64[::1, :]` |

> **一句话记住**：访问顺序要和内存顺序一致，让 CPU 顺着"一维地址线"往前读，不要跳来跳去。

### 3.3.4 查看数组的类型和布局

```python
arr = np.ones((3, 4), dtype=np.float32, order='C')
print(numba.typeof(arr))
# Array(float32, 2, 'C', readonly=False)

arr_f = np.asfortranarray(arr)
print(numba.typeof(arr_f))
# Array(float32, 2, 'F', readonly=False)
```

### 3.3.5 确保数组布局兼容

```python
@njit
def process_c_order(arr):
    return arr + 1

# 传入 F-contiguous 数组
arr_f = np.asfortranarray(np.ones(100, dtype=np.float64))
result = process_c_order(arr_f)  # ✅ Numba 自动处理（可能产生拷贝）

# 更好的做法：调用前转换
arr_c = np.ascontiguousarray(arr_f)
result = process_c_order(arr_c)  # 无拷贝，直接传 C-contiguous
```

---

## 3.4 复合类型

### 3.4.1 元组 (Tuple)

Numba 支持两种元组：

**同构元组 (UniTuple)**：所有元素类型相同

```python
from numba import types

# 三个 float64 的元组
types.UniTuple(types.float64, 3)   # 或简写：float64 × 3
```

```python
@njit
def process_tuple(t):
    return t[0] + t[1] + t[2]

process_tuple((1.0, 2.0, 3.0))     # ✅ 同构元组 (float64 × 3)
process_tuple((1, 2.0, 3))         # ❌ 异构元组 → 可能需要 cast
```

**异构元组 (Tuple)**：元素类型可以不同

```python
@njit
def process_mixed():
    t = (1, 2.0, "hello")          # (int64, float64, unicode_type)
    return t[0], t[1], len(t[2])
```

### 3.4.2 列表 (List)

⚠️ Numba 的列表与 Python 列表有本质区别：**元素类型必须在编译时确定**。

从 Numba 0.45+ 开始，推荐使用 `numba.typed.List`：

```python
from numba.typed import List

@njit
def list_demo():
    # 创建空 typed list（必须在创建时指定元素类型）
    lst = List.empty_list(types.float64)
    # 或者通过初始值推断
    lst = List([1.0, 2.0, 3.0])    # 推断为 List(float64)
    
    lst.append(4.0)
    lst.append(5.0)
    return lst
```

**typed list 的限制**：
- 不支持切片赋值 `lst[1:3] = ...`
- 不支持 `lst.sort()`、`lst.reverse()`
- `pop()` 不带参数有效，带索引可能有问题
- 嵌套 `typed.List` 需要显式声明类型

### 3.4.3 字典 (Dict)

```python
from numba.typed import Dict

@njit
def dict_demo():
    # 通过初始值推断
    d = Dict()
    d['a'] = 1
    d['b'] = 2
    
    # 显式类型
    d2 = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.int64
    )
    return d
```

**typed dict 的限制**：
- 不支持 `dict.pop(key, default)`
- `.keys()` / `.values()` / `.items()` 返回视图，可遍历
- 不支持字典推导式

### 3.4.4 字符串

Numba 支持有限的字符串操作：

| 支持 | 不支持 |
|------|--------|
| `+` 拼接 | `.format()` |
| `len()` | `.split()` |
| 比较 (`==`, `!=`, `<`, `>`) | `.strip()` |
| 索引访问 `s[i]` | 正则 |
| 遍历（逐字符） | `.upper()` / `.lower()` |

---

## 3.5 特殊类型

### 3.5.1 可选类型 (Optional)

```python
from numba import types, njit

# 一个既可以是 None 也可以是 int64 的变量
@njit
def maybe_add(x):
    result = 0
    if x > 10:
        result = None   # ❌ result 已经是 int64，不能变成 None
    return result

# 正确做法：用可选类型
from numba.typed import List

@njit
def optional_example(arr, default_value=None):
    # 如果参数可能为 None，Numba 会自动推断为可选类型
    if default_value is None:
        return np.mean(arr)
    return np.mean(arr) + default_value
```

### 3.5.2 函数类型

```python
@njit
def square(x):
    return x * x

@njit
def apply_func(arr, func):
    """func 被推断为函数类型，可以传入 njit 函数"""
    result = np.empty_like(arr)
    for i in range(len(arr)):
        result[i] = func(arr[i])
    return result

result = apply_func(np.array([1.0, 2.0, 3.0]), square)
```

### 3.5.3 void 类型

```python
@njit(types.void(types.int64))
def print_number(n):
    print("Number:", n)    # 仅用于调试，无返回值
```

---

## 3.6 类型反射 —— 排查类型问题的利器

### 3.6.1 `numba.typeof()`

```python
import numpy as np
from numba import typeof, types

arr = np.zeros((3, 4), dtype=np.int32, order='F')
print(typeof(arr))
# Array(int32, 2, 'F', readonly=False)

print(typeof(42))
# int64

print(typeof(3.14))
# float64
```

### 3.6.2 `inspect_types()`

查看函数内每个变量的类型推断结果：

```python
@njit
def complex_func(a, b):
    c = a + b
    if c > 10:
        d = c * 2.0
    else:
        d = c * 0.5
    return d

complex_func(5.0, 3.0)
print(complex_func.inspect_types())
```

这会输出类似：

```
complex_func(int64, int64)
  c = a + b  :: int64
  c > 10     :: bool
  d = c * 2.0 :: float64
  d = c * 0.5 :: float64
  return d   :: float64
```

### 3.6.3 查看已编译的签名

```python
@njit
def add(a, b):
    return a + b

add(1, 2)
add(1.0, 2.0)
print(add.signatures)
# [(int64, int64), (float64, float64)]
```

---

## 3.7 常见类型错误与排查

| 错误信息 | 原因 | 解决 |
|---------|------|------|
| `Cannot unify int64 and float64` | 同一变量在不同分支被推断为不同类型 | 在分支前初始化变量为统一类型 |
| `No conversion from List[object] to Array` | 传了 Python list 给需要数组的函数 | 传入前转为 `np.array()` |
| `reflected list` 警告 | 用了 Python 原生 list 而非 typed list | 改用 `numba.typed.List` |
| `Can't infer type of variable 'x'` | 变量没有被赋初始值或类型信息不够 | 给变量赋初始值，或用显式类型签名 |
| `Array type mismatch: C vs F` | 布局不匹配（较新版本会自动处理） | 用 `np.ascontiguousarray` 确保布局 |

---

## 3.8 本章关键概念

| 概念 | 一句话 |
|------|--------|
| **类型推断** | Numba 从输入参数出发，追踪每个变量的类型直到 return |
| **统一 (Unification)** | 同一变量在所有路径上类型必须一致 |
| **C-contiguous** | NumPy 默认布局，最后维度在内存中连续 |
| **F-contiguous** | 列优先布局，第一维度在内存中连续 |
| **typed list/dict** | Numba 中专用的类型固定容器（`numba.typed`），不同于 Py 原生 |
| **`typeof()`** | 返回 Numba 视角下的变量类型 |
| **`inspect_types()`** | 查看编译后每个变量的推断类型 |
| **可选类型** | 允许变量为 `None` 或某种具体类型 |

---

## 3.9 下章预告

第四章将讲解 Numba 中**支持的 Python 特性**：控制流、内置函数、数据结构、闭包、递归——以及那些常见的"坑"。
