# 第七章：ufunc 向量化加速 —— `@vectorize` 和 `@guvectorize`

> **核心问题**：`@njit` 加速循环，但每次还是要手写循环。能不能写一个对标量操作的函数，让 Numba 自动给你套上循环、广播、类型转换——就像 NumPy 的 `np.sin` 那样？

---

## 7.1 什么是 ufunc，为什么它快

### 7.1.1 NumPy ufunc 回顾

你每天都在用 ufunc（universal function），只是可能没意识到：

```python
arr = np.array([1.0, 2.0, 3.0, 4.0])

# 这些都是 ufunc：
np.sqrt(arr)     # → sqrt([1, 2, 3, 4]) = [1, 1.414, 1.732, 2]
np.sin(arr)      # → sin([1, 2, 3, 4])
arr + 1          # → [2, 3, 4, 5]  （加法也是 ufunc！）
```

ufunc 的特点：**接受标量或数组，自动对每个元素执行相同操作**。你不需要写 `for` 循环。

### 7.1.2 Numba 的 ufunc 能力

Numba 的 `@vectorize` 和 `@guvectorize` 让你**自己创建 ufunc**。写一个对标量的函数，装饰后自动获得：

- 🚀 **自动循环** —— 不需要手写 for
- 📡 **自动广播** —— 像 NumPy 一样自动处理不同形状的数组
- 🔄 **自动类型分发** —— 传 `float32` 就跑 `float32` 版本，传 `float64` 就跑 `float64` 版本
- ⚡ **可选并行** —— `target='parallel'` 一步启用多核

```python
from numba import vectorize

# 你只写对标量的操作
@vectorize
def clamp(x, lo, hi):
    if x < lo: return lo
    if x > hi: return hi
    return x

# Numba 自动把它"升维"成 ufunc
arr = np.array([-1, 5, 3, 10, 0])
result = clamp(arr, 0, 8)   # → [0, 5, 3, 8, 0]
#          ↑ 每个元素都被 clamp 处理了，你没写一个 for 循环
```

---

## 7.2 `@vectorize` —— 把标量函数变成 ufunc

### 7.2.1 基本用法

```python
from numba import vectorize
import numpy as np

@vectorize
def relu(x):
    return max(0, x)

arr = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
print(relu(arr))  # [0, 0, 0, 1, 2]
```

一行装饰器，一个对标量写的函数——Numba 帮你做了剩下所有事：遍历数组元素、类型判断、返回新数组。

### 7.2.2 指定输入输出类型（签名）

不带签名时，Numba 在第一次调用时做类型推断（和 `@njit` 一样）。你也可以显式指定：

```python
# 单个签名：输入 float64, float64 → 输出 float64
@vectorize('float64(float64, float64)')
def weighted_sum(x, y):
    return x * 0.7 + y * 0.3

# 多个签名：支持多组类型组合
@vectorize(['float64(float64, float64)',
            'float32(float32, float32)',
            'int64(int64, int64)'])
def add(a, b):
    return a + b

# 用 Numba 类型对象（等同于上面）
from numba import float64, float32, int64
@vectorize([float64(float64, float64),
            float32(float32, float32),
            int64(int64, int64)])
def add_typed(a, b):
    return a + b
```

**显式签名的好处**：
- 跳过类型推断，编译更快
- 精确控制每个类型版本的行为
- 文档作用——一看签名就知道函数接受什么类型

### 7.2.3 多输入 / 多输出

```python
# 多输入
@vectorize('float64(float64, float64, float64)')
def lerp(a, b, t):
    """线性插值"""
    return a + (b - a) * t

# 不支持多输出 —— 如果需要返回多个值，改用 @guvectorize
```

`@vectorize` 只能返回**一个值**。如果需要返回多个数组（如 (商, 余数)、均值+标准差同时输出），请用下一节的 `@guvectorize`。

### 7.2.4 `target` 参数 —— 一键切换硬件后端

```python
# 单核 CPU（默认）
@vectorize('float64(float64)', target='cpu')
def cpu_func(x):
    return x * x + 1

# 多核 CPU 并行
@vectorize('float64(float64)', target='parallel')
def parallel_func(x):
    return x * x + 1

# GPU（需 numba-cuda）
@vectorize('float64(float64)', target='cuda')
def gpu_func(x):
    return x * x + 1
```

| target | 说明 | 适用场景 |
|--------|------|---------|
| `'cpu'` | 单核顺序执行 | 默认，小数组 (< 10³) |
| `'parallel'` | 多核并行 | 大数组 (> 10⁴)，CPU 密集 |
| `'cuda'` | GPU 执行 | 超大数组 (> 10⁶)，需要 NVIDIA GPU + numba-cuda |

---

## 7.3 `@guvectorize` —— 广义 ufunc

### 7.3.1 `@vectorize` 的局限

`@vectorize` 只能处理**一个元素一个元素**的操作。但很多算法需要"看"周围元素或处理子数组：

- 滑动窗口平均 —— 需要对一小段子数组操作
- 矩阵行归一化 —— 需要对每行做操作（输入整行，输出整行）
- 自定义距离计算 —— 输入两个向量，输出一个标量

这些情况下 `@vectorize` 不够用，你需要 `@guvectorize`（generalized universal function）。

### 7.3.2 `@guvectorize` 的基本模式

```python
from numba import guvectorize, float64

@guvectorize('(n), (n) -> ()',      # 签名：两个一维数组 → 一个标量
             'float64[:,:], float64[:,:], float64[:]')
             # ↑ 实际传入的是矩阵，但每"行"被当作 (n) 来处理
def row_cosine_similarity(a, b, out):
    """计算两行之间的余弦相似度"""
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for i in range(len(a)):
        dot += a[i] * b[i]
        norm_a += a[i] * a[i]
        norm_b += b[i] * b[i]
    out[0] = dot / (np.sqrt(norm_a) * np.sqrt(norm_b))
```

`@guvectorize` 最反直觉的地方是**输出参数放在参数列表最后**，而不是通过 `return`。这是因为 Numba 需要把输出数组预分配好，然后让函数往里写。

### 7.3.3 布局声明语法详解

`@guvectorize` 的签名有三部分，用逗号分隔：

```python
@guvectorize('(输入维度声明) -> (输出维度声明)',     # 核心维度
             '输入布局, 输入布局, 输出布局')           # 内存布局
```

**核心维度声明**：

```python
(m), (n) -> (m, n)       # 两个一维 → 一个二维
(n), (n) -> ()           # 两个一维 → 一个标量
(n, m), (m, p) -> (n, p) # 两个二维 → 矩阵乘
(n) -> (n)               # 一维 → 一维（元素级变换）
```

**内存布局声明**（第二个参数）：

```python
float64[:]         # 一维 float64
float64[:, :]      # 二维 float64
float64[:, :, :]   # 三维 float64
```

**完整示例**：

```python
@guvectorize('(n), (n) -> (n)',    # 核心：两个等长一维 → 一维
             'float64[:], float64[:], float64[:]')  # 布局
def multiply_each(a, b, out):
    for i in range(len(a)):
        out[i] = a[i] * b[i]
```

### 7.3.4 `@vectorize` vs `@guvectorize` 对比

| | `@vectorize` | `@guvectorize` |
|--|-------------|---------------|
| **粒度** | 逐元素 (element-wise) | 子数组 (sub-array) |
| **输出** | `return` 返回值 | 写最后一个参数 `out` |
| **核心维度** | 不需要声明 | 需要 `(m),(n)->(...)` 声明 |
| **适用场景** | 激活函数、逐元素运算、裁剪 | 窗口计算、行/列操作、距离/相似度 |
| **并行** | `target='parallel'` | 不支持 `target`，但支持 `parallel=True` (较新版本) |

### 7.3.5 实战：滑动窗口均值

```python
@guvectorize('(n), () -> (n)',
             'float64[:], int64, float64[:]')
def moving_average(x, window, out):
    """滑动窗口均值 —— 不能逐元素做，需要看周围元素"""
    n = len(x)
    half = window // 2
    for i in range(n):
        total = 0.0
        count = 0
        for j in range(max(0, i - half), min(n, i + half + 1)):
            total += x[j]
            count += 1
        out[i] = total / count
```

---

## 7.4 机器学习中的实战示例

### 7.4.1 自定义激活函数

```python
@vectorize('float64(float64)')
def swish(x):
    """Swish 激活函数：x * sigmoid(x)，Google 提出"""
    return x / (1 + np.exp(-x))
```

### 7.4.2 距离矩阵

```python
@guvectorize('(n), (n) -> ()',
             'float64[:], float64[:], float64[:]')
def euclidean_distance(a, b, out):
    """两个向量之间的欧氏距离"""
    s = 0.0
    for i in range(len(a)):
        diff = a[i] - b[i]
        s += diff * diff
    out[0] = np.sqrt(s)

# 使用：矩阵中每一对行之间的距离
X = np.random.randn(100, 10)   # 100 个样本，每个 10 维
# guvectorize 自动对每一行/每一行组合调用函数
distances = euclidean_distance(X[:, None, :], X[None, :, :])
# 返回 100×100 的距离矩阵
```

### 7.4.3 图像处理

```python
@guvectorize('(n, n) -> (n, n)',
             'float64[:, :], float64[:, :]')
def sobel_edge(y, out):
    """Sobel 边缘检测核（简化版）"""
    m, n = y.shape
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            gx = -y[i-1, j-1] + y[i-1, j+1] \
                 -2*y[i, j-1] + 2*y[i, j+1] \
                 -y[i+1, j-1] + y[i+1, j+1]
            gy = -y[i-1, j-1] - 2*y[i-1, j] - y[i-1, j+1] \
                 + y[i+1, j-1] + 2*y[i+1, j] + y[i+1, j+1]
            out[i, j] = np.sqrt(gx*gx + gy*gy)
```

---

## 7.5 性能对比：手写循环 vs @vectorize vs @guvectorize

```python
import numpy as np
import time
from numba import njit, vectorize, guvectorize

# 手写循环
@njit
def add_loop(a, b):
    result = np.empty_like(a)
    for i in range(len(a)):
        result[i] = a[i] + b[i]
    return result

# @vectorize
@vectorize('float64(float64, float64)')
def add_vec(a, b):
    return a + b

# @guvectorize
@guvectorize('(n), (n) -> (n)',
             'float64[:], float64[:], float64[:]')
def add_guvec(a, b, out):
    for i in range(len(a)):
        out[i] = a[i] + b[i]

# 三者对逐元素加法性能几乎一致
# @vectorize 的优势是代码量最少，且 target='parallel' 无需手动 prange
# @guvectorize 的优势是能做子数组操作（窗口、行列等）
```

> **规则**：
> - 元素级操作 → `@vectorize`
> - 窗口/行/列/子数组操作 → `@guvectorize`
> - 复杂逻辑、多函数协作 → `@njit`

---

## 7.6 本章关键概念

| 概念 | 一句话 |
|------|--------|
| **ufunc** | 对标量写代码，自动应用到数组每个元素 |
| **`@vectorize`** | 创建元素级 ufunc，写一个对标量的函数即可 |
| **`@guvectorize`** | 创建子数组级 ufunc，支持窗口/行列操作 |
| **核心维度** | `(n),(n)->()` —— 声明输入/输出的形状关系 |
| **输出在参数里** | `@guvectorize` 用最后一个参数 `out` 写结果，不是 return |
| **`target`** | `'cpu'` / `'parallel'` / `'cuda'`，一键切换硬件 |
| **选型** | 逐元素→`@vectorize`，窗口/行列→`@guvectorize`，复杂→`@njit` |

---

## 7.7 下章预告

第八章进入**并行计算**——`parallel=True`、`prange`、线程控制、竞态条件与归约。
