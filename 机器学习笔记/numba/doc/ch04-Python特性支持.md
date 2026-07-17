# 第四章：支持的 Python 特性

> **核心问题**：Numba 的 `@njit` 不是全能 Python 编译器——它只编译 Python 和 NumPy 的一个子集。哪些能写、哪些不能写？本章一次性讲清楚。

---

## 4.1 控制流

好消息：**绝大部分控制流都支持**，写法和你熟悉的 Python 完全一样。

### 4.1.1 `if` / `elif` / `else`

✅ 完全支持，但**所有分支中同一变量的类型必须统一**（第三章讲过）：

```python
@njit
def grade(score):
    if score >= 90:
        level = 'A'
    elif score >= 80:
        level = 'B'
    else:
        level = 'C'
    return level       # ✅ level 在所有分支中都是字符串，类型统一
```

### 4.1.2 `for` 循环

✅ 完全支持：

```python
@njit
def for_examples(arr):
    # range 循环（整数参数）
    for i in range(len(arr)):        # ✅
        arr[i] = arr[i] ** 2

    # 直接迭代数组
    for x in arr:                     # ✅ 遍历一维数组
        ...

    # 逐个元素（不能用 enumerate 配合二维数组的多维索引）
    for i in range(arr.shape[0]):     # ✅ 手动管理二维遍历
        for j in range(arr.shape[1]):
            arr[i, j] += 1
```

### 4.1.3 `while` 循环

✅ 支持，包括无限循环：

```python
@njit
def find_first_negative(arr):
    i = 0
    while i < len(arr):
        if arr[i] < 0:
            return i
        i += 1
    return -1
```

### 4.1.4 `break` / `continue` / `return`

✅ 全部支持，语义和 Python 一致。

---

## 4.2 内置函数

| 函数 | 支持 | 备注 |
|------|:--:|------|
| `abs(x)` | ✅ | 整数、浮点、复数 |
| `min(a, b)` / `max(a, b)` | ✅ | 两个参数版本 |
| `sum(iterable)` | ✅ | 仅支持迭代数组/列表 |
| `len(x)` | ✅ | 数组、列表、元组、字符串 |
| `range(start, stop, step)` | ✅ | 参数必须是整数 |
| `enumerate(seq)` | ✅ | 遍历数组/列表 |
| `zip(a, b)` | ✅ | 同类型数组/列表 |
| `round(x, n)` | ✅ | 四舍五入到 n 位小数 |
| `print(...)` | ✅ | **仅用于调试**，会拖慢性能 |
| `bool(x)` / `int(x)` / `float(x)` | ✅ | 类型转换 |
| `isinstance(x, type)` | ✅ | 检查 Numba 类型 |
| `type(x)` | ⚠️ | 返回 Numba 类型对象，不是 Python `type` |
| `hash(x)` | ❌ | 不支持 |

**示例**：

```python
@njit
def builtin_demo(arr):
    # min/max
    smallest = min(arr[0], arr[1])

    # enumerate
    for idx, val in enumerate(arr):
        if val < 0:
            return idx

    # zip
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    for x, y in zip(a, b):
        print(x + y)

    # round
    return round(3.14159, 2)   # 3.14
```

---

## 4.3 Python 数据结构

### 4.3.1 元组 (Tuple)

✅ 支持，**创建后不可变**（和 Python 一致）：

```python
@njit
def tuple_demo():
    t = (1, 2.0, "hello")       # 异构元组：类型自动推断
    a = t[0]                      # ✅ 按索引访问
    b, c, d = t                   # ✅ 解包

    # 同构元组（所有元素类型相同）
    t2 = (1.0, 2.0, 3.0)         # 推断为 UniTuple(float64, 3)
    return t2[0]

# 也可以在函数签名中接收元组
@njit
def process(t):
    return t[0] + t[1]

process((1.0, 2.0))              # ✅ 传入 Python 元组，Numba 自动识别
```

### 4.3.2 列表

⚠️ **普通 Python `list` 不能直接在 `@njit` 中使用**——你需要用 `numba.typed.List`（第三章 §3.4.2 已讲）。如果你把普通 Python list 传给 `@njit` 函数，Numba 会发出 `reflected list` 警告，性能很差。

```python
from numba.typed import List

# ✅ 正确做法：在 Numba 外创建 typed list，传入 njit 函数
lst = List()
lst.append(1.0)
lst.append(2.0)
my_njit_func(lst)                # 直接传入

# ✅ 或者在 njit 函数内部创建
@njit
def inner_list_demo():
    lst = List.empty_list(types.float64)
    lst.append(1.0)
    return lst
```

### 4.3.3 字典

⚠️ 同样必须用 `numba.typed.Dict`：

```python
from numba.typed import Dict

d = Dict()
d['x'] = 10
my_njit_func(d)
```

### 4.3.4 集合 (Set)

❌ **Numba 不支持 Python 的 `set`**。没有对应的 typed 容器，需要改用其他方案（如排序后二分查找、用 typed dict 的 key 来模拟）。

---

## 4.4 函数与闭包

### 4.4.1 嵌套函数

✅ 支持内部函数定义：

```python
@njit
def outer(a, b):
    def inner(x):
        return x * x + 1
    return inner(a) + inner(b)
```

**限制**：嵌套函数不能被外部访问（它的类型是编译时确定的，不能作为返回值传给 `@njit` 之外的代码）。

### 4.4.2 闭包

✅ 内部函数可以捕获外部变量：

```python
@njit
def make_adder(increment):
    def add(x):                  # ✅ 捕获了 increment
        return x + increment
    return add(10)               # ✅ 在 @njit 内部调用没问题
```

### 4.4.3 递归

✅ **有限支持**。递归深度不能太大（没有 Python 的递归深度保护），且递归函数不能是闭包：

```python
@njit
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)   # ✅ 递归调用自身

factorial(10)  # ✅
factorial(1000)  # ⚠️ 可能栈溢出——Numba 没有递归深度限制
```

---

## 4.5 不支持的特性（常见坑）

这是每个 Numba 新手都会撞上的问题，提前知道可以省你很多时间。

### 4.5.1 Python 标准库 —— 几乎全部不能用

```python
@njit
def bad_idea():
    import json                    # ❌ json 模块不能导入
    import re                      # ❌ re 模块不能导入
    from datetime import datetime  # ❌ datetime 不能导入
    data = open('file.txt')        # ❌ 文件 I/O 不能
    url = requests.get('...')      # ❌ 网络不能
    ...
```

**规则**：`@njit` 函数里**不能 `import` 任何 Python 模块**（除了 `numpy` 和 `numba` 自身）。

### 4.5.2 `try` / `except` / `raise`

⚠️ 有限支持：

```python
# ✅ 支持：简单的 raise
@njit
def divide(a, b):
    if b == 0:
        raise ValueError("除数不能为零")
    return a / b

# ❌ 不支持：try/except 捕获异常
@njit
def bad_catch(a, b):
    try:
        return a / b
    except ZeroDivisionError:      # ❌ 编译失败！
        return 0

# ⚠️ Numba 0.50+ 部分支持 try/except，但限制很多
# 建议：用 if/else 提前检查条件来替代 try/except
```

### 4.5.3 生成器 `yield`

❌ **不支持**。Numba 不能编译包含 `yield` 的函数：

```python
# ❌ 这不能编译
@njit
def bad_generator(n):
    for i in range(n):
        yield i * i

# ✅ 替代方案：用数组收集结果后返回
@njit
def good_array(n):
    result = np.empty(n, dtype=np.int64)
    for i in range(n):
        result[i] = i * i
    return result
```

### 4.5.4 `*args` 和 `**kwargs`

❌ **不支持可变参数**：

```python
@njit
def bad_args(*args):              # ❌ 不支持
    ...

@njit
def bad_kwargs(**kwargs):         # ❌ 不支持
    ...

# ✅ 用固定参数
@njit
def good_func(a, b, c=0):        # ✅ 支持默认参数
    return a + b + c
```

### 4.5.5 动态属性与 `getattr` / `setattr`

❌ 不支持。Numba 必须在编译时确定所有属性：

```python
class MyClass:
    pass
obj = MyClass()
obj.dynamic_name = 123             # ❌ Numba 中不支持，类需要用 @jitclass
```

### 4.5.6 其他不支持项速查

| 特性 | 状态 | 替代方案 |
|------|:----:|---------|
| `lambda` | ❌ | 写成嵌套函数 |
| 列表推导式 / 字典推导式 | ❌ | 用 `for` 循环 + `append` |
| `with` 语句 | ❌ | 手动管理（罕见需要） |
| `del` 删除变量 | ❌ | 变量自然超出作用域 |
| `slice` 对象 | ❌ | 用 `range(start, stop, step)` |
| `eval` / `exec` | ❌ | — |
| `globals()` / `locals()` | ❌ | — |

---

## 4.6 如何验证某个特性是否可用

**方法一：直接试**

```python
@njit
def test():
    # 写你想试的代码
    ...

test()   # 如果抛出 TypingError，就是不支持
```

**方法二：查官方支持列表**

Numba 官方文档有一个完整的「Supported Python features」页面，列出了所有支持的语法和内置函数。遇到不确定的，直接搜索 `numba supported python features`。

**方法三：用 `inspect_types()` 诊断**

养成习惯：写完 `@njit` 函数后跑一遍 `inspect_types()`，确认所有代码都编译成功，没有"回退"到 object 模式。

---

## 4.7 本章关键概念

| 概念 | 一句话 |
|------|--------|
| **支持的控制流** | `if/for/while/break/continue/return` —— 和 Python 一模一样 |
| **支持的序列操作** | `range/enumerate/zip` —— 够用但不全 |
| **typed list/dict** | 必须用 `numba.typed` 版本，不能传普通 Python 容器 |
| **标准库 = 零** | 不能 `import` 任何第三方/Python 标准库模块 |
| **try/except 有限** | 简单 `raise` 可以，`try/except` 支持不完整，建议用 if 提前判断 |
| **yield = 不行** | 用数组收集结果代替生成器 |
| **lambda = 不行** | 写成嵌套函数 |
| **每写一个函数就 `inspect_types()`** | 确认编译成功，没有回退 |

---

## 4.8 下章预告

第五章将深入 Numba 中**支持的 NumPy 特性**——哪些函数可以加速、哪些不行、以及那些容易踩的坑。
