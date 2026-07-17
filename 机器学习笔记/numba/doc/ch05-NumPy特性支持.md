# 第五章：支持的 NumPy 特性

> **核心问题**：Numba 号称加速 NumPy，但它到底支持哪些 NumPy 函数？哪些不支持？支持的和原生 NumPy 又有什么区别？

---

## 5.1 总览：Numba 中的 NumPy 支持范围

Numba 对 NumPy 的支持可以分为四个层级：

| 层级 | 说明 | 例子 |
|------|------|------|
| ⭐ **完全支持** | 行为和 NumPy 一致，性能更好 | `np.sum`、`np.dot`、数组索引 |
| ⚠️ **部分支持** | 大部分功能可用，但有细节差异 | `np.linalg.svd`（只返回核心 3 个值） |
| 🔧 **Numba 替代** | NumPy 版本不能用，Numba 提供了自己的版本 | `numba.random` 代替 `np.random` |
| ❌ **不支持** | 没有替代方案 | 高级索引（花式索引部分限制） |

---

## 5.2 数组创建

### 5.2.1 常用创建函数

| 函数 | 支持 | 备注 |
|------|:--:|------|
| `np.zeros(shape)` | ✅ | 和 NumPy 一样 |
| `np.ones(shape)` | ✅ | |
| `np.empty(shape)` | ✅ | 未初始化内存，速度最快 |
| `np.zeros_like(arr)` | ✅ | |
| `np.ones_like(arr)` | ✅ | |
| `np.empty_like(arr)` | ✅ | |
| `np.arange(start, stop, step)` | ✅ | |
| `np.linspace(start, stop, n)` | ✅ | |
| `np.eye(n)` | ✅ | 单位矩阵 |
| `np.identity(n)` | ✅ | |
| `np.diag(arr)` | ✅ | |
| `np.full(shape, value)` | ✅ | |
| `np.full_like(arr, value)` | ✅ | |
| `np.copy(arr)` | ✅ | 深拷贝 |
| `np.array(...)` | ✅ | 从列表创建数组 |
| `np.asarray(...)` | ✅ | |

```python
@njit
def create_arrays(n):
    a = np.zeros(n, dtype=np.float64)        # [0, 0, 0, ...]
    b = np.ones(n, dtype=np.float64)         # [1, 1, 1, ...]
    c = np.empty(n, dtype=np.float64)        # 垃圾值，使用时需先赋值
    d = np.arange(n, dtype=np.float64)       # [0, 1, 2, ..., n-1]
    e = np.linspace(0, 1, n)                 # 均匀分布的 n 个数
    f = np.eye(3)                             # 3×3 单位矩阵
    g = np.full(n, 42.0)                     # [42, 42, 42, ...]
    return c  # 需要在返回前给 c 赋值！
```

### 5.2.2 数组创建的注意事项

```python
# ❌ 不能在 @njit 内用 Python list 来构造 np.array
@njit
def bad():
    arr = np.array([1, 2, 3])          # ⚠️ 3 个以内的短列表通常可以，但不可靠
    arr2 = np.array([i for i in range(100)])  # ❌ 列表推导式不支持

# ✅ 用 np.arange 或先 empty 再填充
@njit
def good(n):
    arr = np.arange(n, dtype=np.float64)
    return arr
```

---

## 5.3 数组运算

### 5.3.1 元素级运算

✅ **全部支持**，写法和 NumPy 一模一样：

```python
@njit
def element_wise(a, b):
    c = a + b          # 加法
    d = a - b          # 减法
    e = a * b          # 元素乘（不是矩阵乘！）
    f = a / b          # 除法
    g = a ** 2         # 幂
    h = a % 3          # 取模
    i = a > b          # 比较 → 布尔数组
    j = (a > 0) & (b < 10)  # 逻辑与（注意用 &，不是 and）
    k = ~i             # 逻辑非
    return c, i
```

> ⚠️ Numba 中 `and`/`or` 是 Python 关键字（短路求值），对数组元素级 AND/OR 請用 `&`/`|`。

### 5.3.2 数学函数

| 函数 | 支持 | 函数 | 支持 |
|------|:--:|------|:--:|
| `np.sqrt(x)` | ✅ | `np.abs(x)` | ✅ |
| `np.sin(x)` / `np.cos(x)` / `np.tan(x)` | ✅ | `np.arcsin(x)` / `np.arccos(x)` / `np.arctan(x)` | ✅ |
| `np.exp(x)` | ✅ | `np.log(x)` / `np.log10(x)` | ✅ |
| `np.power(x, y)` | ✅ | `np.mod(x, y)` | ✅ |
| `np.sign(x)` | ✅ | `np.ceil(x)` / `np.floor(x)` | ✅ |
| `np.clip(x, lo, hi)` | ✅ | `np.where(cond, a, b)` | ✅ |
| `np.isnan(x)` | ✅ | `np.isinf(x)` | ✅ |
| `np.maximum(a, b)` | ✅ | `np.minimum(a, b)` | ✅ |

```python
@njit
def math_demo(arr):
    # 所有运算都是元素级的（和 NumPy 完全一致）
    result = np.sqrt(np.abs(arr))       # sqrt(|x|)
    result = np.clip(result, 0, 5)      # 截断到 [0, 5]
    # np.where：类似三元表达式，对每个元素
    cleaned = np.where(np.isnan(arr), 0.0, arr)
    return result, cleaned
```

### 5.3.3 聚合函数

| 函数 | 支持 | 备注 |
|------|:--:|------|
| `np.sum(arr)` | ✅ | |
| `np.sum(arr, axis=0)` | ✅ | 指定轴求和 |
| `np.mean(arr)` | ✅ | |
| `np.std(arr)` | ✅ | |
| `np.var(arr)` | ✅ | |
| `np.min(arr)` / `np.max(arr)` | ✅ | |
| `np.argmin(arr)` / `np.argmax(arr)` | ✅ | |
| `np.prod(arr)` | ✅ | |
| `np.median(arr)` | ❌ | 不支持——需要自己实现 |
| `np.percentile(arr, q)` | ❌ | 不支持——需要自己实现 |

```python
@njit
def stats_demo(arr):
    s = np.sum(arr)
    m = np.mean(arr)
    # 按列求和（axis=0）
    col_sums = np.sum(arr, axis=0)
    return s, m, col_sums
```

---

## 5.4 数组操作

### 5.4.1 索引与切片

```python
@njit
def indexing_demo(arr):
    x = arr[0]              # ✅ 标量索引
    y = arr[1:5]            # ✅ 切片
    z = arr[1:10:2]         # ✅ 带步长切片
    w = arr[-1]             # ✅ 负数索引
    arr[0] = 99             # ✅ 原地修改
    return x, y, z, w

@njit
def fancy_indexing(arr, indices):
    # 花式索引（整数数组索引）
    result = arr[indices]   # ✅ 一维花式索引
    return result
```

### 5.4.2 形状与布局操作

| 操作 | 支持 | 备注 |
|------|:--:|------|
| `arr.shape` | ✅ | 返回元组 |
| `arr.ndim` | ✅ | |
| `arr.size` | ✅ | |
| `arr.dtype` | ✅ | |
| `np.reshape(arr, newshape)` | ✅ | |
| `arr.reshape(newshape)` | ✅ | |
| `arr.ravel()` | ✅ | 展平为一维 |
| `arr.flatten()` | ✅ | 展平（副本） |
| `np.transpose(arr)` | ✅ | |
| `arr.T` | ✅ | |
| `np.moveaxis(arr, src, dst)` | ✅ | 0.64+ 新增 |
| `np.concatenate((a, b))` | ✅ | |
| `np.stack((a, b))` | ⚠️ | 部分支持 |
| `np.column_stack` / `np.row_stack` | ❌ | 用 concatenate 替代 |

### 5.4.3 广播 (Broadcasting)

✅ Numba 完全支持 NumPy 的广播机制：

```python
@njit
def broadcast_demo():
    a = np.ones((3, 4))     # (3, 4)
    b = np.array([1, 2, 3, 4])  # (4,)
    c = a + b               # ✅ 广播：b 自动扩展为 (3, 4)
    return c
```

### 5.4.4 矩阵乘法

```python
@njit
def matrix_ops(A, B):
    # 矩阵乘法
    C = np.dot(A, B)        # ✅ 二维矩阵乘
    D = A @ B               # ✅ Python 3.5+ 运算符（仅 Numba 0.59+）
    
    # 一维向量点积
    u = np.array([1.0, 2.0, 3.0])
    v = np.array([4.0, 5.0, 6.0])
    dot_uv = np.dot(u, v)   # ✅ 标量结果
    
    return C, dot_uv
```

---

## 5.5 线性代数 (`np.linalg`)

Numba 支持大部分常用的线性代数函数，但**并非所有参数都支持**：

| 函数 | 支持 | 备注 |
|------|:--:|------|
| `np.linalg.inv(A)` | ✅ | 方阵求逆 |
| `np.linalg.det(A)` | ✅ | 行列式 |
| `np.linalg.solve(A, b)` | ✅ | 解 Ax = b |
| `np.linalg.eig(A)` | ✅ | 特征值和特征向量 |
| `np.linalg.eigh(A)` | ✅ | 对称矩阵的特征值 |
| `np.linalg.svd(A)` | ⚠️ | 只返回 U, S, V（不全） |
| `np.linalg.qr(A)` | ✅ | QR 分解 |
| `np.linalg.norm(x)` | ✅ | |
| `np.linalg.cholesky(A)` | ✅ | Cholesky 分解 |

```python
@njit
def linalg_demo(A, b):
    x = np.linalg.solve(A, b)    # 解线性方程组
    detA = np.linalg.det(A)      # 行列式
    return x, detA
```

> ⚠️ `np.linalg.svd` 在 Numba 中只返回标准的 `(U, S, V)` 三元组，不支持 `full_matrices=False` 等参数。

---

## 5.6 随机数生成 —— 深入 Numba 的独立 RNG 系统

> ⚠️ 这是 Numba 中最容易产生"隐性 bug"的模块。写法和 NumPy 一模一样，底层却是完全独立的两套随机数引擎。

### 5.6.1 写的一样，跑的完全不是一回事

先看一段代码，猜猜输出什么：

```python
import numpy as np

# 在外部 Python 中设种子
np.random.seed(42)
a = np.random.randn(3)

@njit
def numba_random():
    return np.random.randn(3)

b = numba_random()

print("外部 NumPy:", a)
print("Numba 内部:", b)
# 输出：两个完全不同的序列！
```

**为什么不一样？** 因为 `@njit` 函数里的 `np.random` 和外面的 `np.random` **走的是两块不同的代码、用不同的随机数状态、甚至用不同的生成算法**。

### 5.6.2 两套 RNG 系统的架构对比

```
┌─────────────────────────────────┐
│       外部 Python 世界           │
│                               │
│  np.random.seed(42)           │
│       │                       │
│       ▼                       │
│  ┌──────────────────┐         │
│  │ NumPy BitGenerator│         │
│  │  (默认: PCG64)   │         │
│  │  全局单例状态     │         │
│  └──────────────────┘         │
│       │                       │
│       ▼                       │
│  np.random.randn() → 结果 A   │
│                               │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│       @njit 函数内部             │
│                               │
│  np.random.seed(42)           │
│       │                       │
│       ▼                       │
│  ┌──────────────────┐         │
│  │ Numba RNG 引擎    │         │
│  │ (xoroshiro128+)  │         │
│  │ 每个线程独立状态   │         │  ← 关键差异！
│  └──────────────────┘         │
│       │                       │
│       ▼                       │
│  np.random.randn() → 结果 B   │
│                               │
└─────────────────────────────────┘
```

| 维度 | 外部 NumPy | Numba `@njit` 内部 |
|------|-----------|-------------------|
| **导入路径** | `numpy.random` | 写 `np.random`，实际走 Numba 重实现 |
| **底层算法** | PCG64 (默认)，可切换为 MT19937、Philox 等 | **xoroshiro128+** (固定不变) |
| **状态存储** | 全局单例 `BitGenerator`，所有调用共享 | **每个线程独立的状态** |
| **种子影响范围** | 影响整个 Python 进程 | 只影响当前线程 |
| **并行行为** | GIL 保护下的串行调用 | 多个线程各自独立生成，互不干扰 |

### 5.6.3 线程级 RNG：每个线程都有自己独立的"骰子"

这是 Numba 随机数最核心的设计：**每个线程拥有自己的 RNG 状态**。

```
并行场景 (prange 或多线程):

线程 0: [seed=42] → xoroshiro128+ → 序列: r00, r01, r02, ...
线程 1: [seed=42+salt] → xoroshiro128+ → 序列: r10, r11, r12, ...
线程 2: [seed=42+2*salt] → xoroshiro128+ → 序列: r20, r21, r22, ...

每个线程的序列相互独立，互不干扰
```

**设计意图**：避免多线程争抢同一个 RNG 状态（那需要加锁，破坏并行性）。

**副作用**（你遇到的问题）：线程调度的顺序是不确定的。今天调度可能是 0→1→2，明天可能是 1→0→2，导致：
- 线程 0 的"第 k 个随机数"可能对应不同的数据
- 如果每个线程负责构建不同部分的特征 → 特征值和线程绑定 → 调度不同 → 结果不同

### 5.6.4 可复现性问题的根因分析

你遇到的粒子滤波 `±0.1 RMSE 抖动`，根因链是这样的：

```
1. 并行特征构建（prange 或 多线程）
      ↓
2. 每个线程用自己的 RNG 状态生成随机特征
      ↓
3. 操作系统线程调度的非确定性
   （这次先跑线程A，下次先跑线程B）
      ↓
4. 线程A用掉 RNG 的"第k个随机数"，但"第k个"对应哪条数据不确定
      ↓
5. 相同的全局种子、相同的代码 → 不同的结果！
      ↓
6. RMSE 出现 ±0.1 的抖动
```

**核心矛盾**：Numba 的线程级 RNG 设计为了性能放弃了确定性——线程之间没有协调，谁能跑就谁消耗随机数。这在 Monte Carlo 模拟、粒子滤波、随机森林等**需要精确可复现**的场景中会出问题。

### 5.6.5 解决方案 1：在 nopython 内核入口显式播种（确定性方案）

**思路**：在函数入口处，用确定的 seed 重置 RNG，使每个线程从相同的起点开始生成，再用线程 ID 做确定性偏移。

```python
@njit
def deterministic_particle_filter(data, global_seed, thread_id):
    """
    解决方案 1: 内核入口显式播种
    - 每个线程用 seed = global_seed + thread_id 初始化
    - 线程调度的顺序不再影响结果（每个线程的 RNG 序列由其 ID 唯一确定）
    """
    # 关键：在函数入口立刻播种，覆盖线程级 RNG 的状态
    np.random.seed(global_seed + thread_id)

    # 后续所有随机数调用都是确定性的
    noise = np.random.randn(len(data))
    # ... 粒子滤波的预测/更新步骤 ...
    return estimate
```

**部署方式**：

```python
# 方式 A: 串行版本（最简单，无需 thread_id）
@njit
def deterministic_serial(data, seed):
    np.random.seed(seed)          # 入口播种
    noise = np.random.randn(len(data))
    # ... 计算 ...
    return result

# 方式 B: 并行版本（prange）
@njit(parallel=True)
def deterministic_parallel(data, seed):
    n = len(data)
    results = np.empty(n)
    for i in prange(n):
        np.random.seed(seed + i)   # 每个迭代用不同的确定性种子
        results[i] = np.random.randn()
    return results
```

**优点**：
- 100% 可复现——同样的 seed → 同样的输出
- 不依赖操作系统线程调度顺序

**代价**：
- 每个函数入口做 `np.random.seed()` 有微小开销（~微秒级）
- 并行版本中，如果 `seed + i` 的设计不当（如相邻迭代的 seed 太接近），可能出现序列相关性（实际影响微乎其微）

### 5.6.6 解决方案 2：多次运行平均的无偏降方差方案

**思路**：既然单次运行受线程调度影响有抖动，那就跑 N 次，用统计方法消除噪声。

```python
def unbiased_estimate(data, n_runs=10):
    """
    解决方案 2: 多次运行平均
    - 每次运行用不同的全局种子
    - 取 N 次结果的平均值 → 方差降为 1/N
    """
    estimates = np.empty(n_runs)
    for run in range(n_runs):
        seed = 42 + run * 1000         # 每次运行不同的种子
        estimates[run] = particle_filter_kernel(data, seed)

    mean_estimate = np.mean(estimates)
    std_estimate = np.std(estimates)   # 量化不确定性
    return mean_estimate, std_estimate
```

**为什么有效**：
- 线程调度的不确定性每次运行产生不同的"偏差方向"
- 多次运行的平均让偏差相互抵消 → **无偏估计**
- 标准差告诉你结果的可靠性

**优点**：
- 不改 `@njit` 函数本身，外部包裹即可
- 天然提供不确定性量化（标准差）
- 和方案 1 互不冲突，可以同时使用

**代价**：
- 运行时间 × N
- 需要确保每次运行用不同的种子（否则 N 次都跑出同一个有偏结果）

### 5.6.7 两种方案的选择指南

| 场景 | 推荐方案 | 原因 |
|------|---------|------|
| 需要精确可复现（论文实验、单元测试） | 方案 1：入口显式播种 | 确定性的，每次都一样 |
| 对可复现性有要求但对性能有要求 | 方案 1 + 串行 | 串行天然确定 |
| 需要并行加速 + 可接受轻微不确定性 | 方案 1 + 并行（seed + thread_id） | 确定性+并行 |
| 需要评估方法稳定性（论文汇报） | 方案 2：多次运行平均 | 给均值+标准差 |
| 生产环境中单次运行结果不可靠 | 方案 1 + 方案 2 组合 | 确定性 + 统计稳健 |

> **最佳实践**：研发/调试阶段用方案 1（确定性，方便复现 bug），论文/汇报阶段用方案 2（汇报均值 ± 标准差）。

### 5.6.8 支持的随机函数

在 `@njit` 内部，以下 `np.random.*` 函数可用：

| 函数 | 支持 | 备注 |
|------|:--:|------|
| `np.random.rand(d0, d1, ...)` | ✅ | [0, 1) 均匀分布 |
| `np.random.randn(d0, d1, ...)` | ✅ | 标准正态分布 |
| `np.random.randint(low, high, size)` | ✅ | 随机整数 |
| `np.random.random(size)` | ✅ | [0, 1) 均匀分布 |
| `np.random.choice(arr, size)` | ✅ | 随机抽样 |
| `np.random.shuffle(arr)` | ✅ | 原地打乱 |
| `np.random.seed(n)` | ✅ | ⚠️ 只在当前线程的 `@njit` 内部生效 |

### 5.6.9 速查：遇到随机数问题怎么办

```
问题：@njit 函数的随机结果每次都不一样？
    ├─→ 你在调用前设了 np.random.seed() 吗？
    │     ├─ 是 → 注意：外部 seed 不影响 Numba 内部
    │     │        必须在 @njit 函数内调用 np.random.seed()
    │     └─ 否 → 在函数入口加 np.random.seed()
    │
    ├─→ 你用了 parallel=True 或 prange 吗？
    │     ├─ 是 → 线程竞争导致不确定性
    │     │        用方案 1（seed+thread_id）或方案 2（多跑取平均）
    │     └─ 否 → 检查是否在循环中创建了新的随机状态
    │
    └─→ 你需要论文级别的可复现性吗？
          是 → 方案 1 + 方案 2 组合
          否 → 方案 1 足够
```

---

## 5.7 NumPy 在 Numba 中的限制和替代方案

### 5.7.1 完全不支持的操作

| 操作 | 替代方案 |
|------|---------|
| `np.median(arr)` | 自己写排序 + 取中位数 |
| `np.percentile(arr, q)` | 自己写排序 + 取分位 |
| `np.histogram(arr)` | 自己写循环统计 |
| `np.unique(arr)` | 自己写排序 + 去重 |
| `np.sort(arr)` | ⚠️ `np.sort()` 本身支持，但原地排序 `arr.sort()` 不支持 |
| `np.loadtxt` / `np.savetxt` | 不支持文件 I/O |
| 大部分 `np.fft.*` | ⚠️ 有限支持，建议用 `scipy.fft` + 外部调用 |
| `np.pad` | 自己写循环填充 |

### 5.7.2 NumPy 2.x 中已移除的函数

这些函数在 NumPy 2.0 中被移除，Numba 0.64 也跟随移除：

| 旧函数 | 替代函数 |
|--------|---------|
| `np.trapz(y, x)` | `np.trapezoid(y, x)` |
| `np.in1d(a, b)` | `np.isin(a, b)` |

### 5.7.3 处理不支持的函数的通用策略

当遇到 Numba 不支持的 NumPy 函数时：

```python
# 策略 1: 拆分成 Numba 部分 + NumPy 部分
@njit
def numba_part(arr):
    # 用 Numba 做我能做的
    sorted_arr = np.sort(arr)
    return sorted_arr

def hybrid_median(arr):
    sorted_arr = numba_part(arr)   # Numba 加速排序
    n = len(sorted_arr)
    if n % 2 == 0:
        return (sorted_arr[n//2-1] + sorted_arr[n//2]) / 2.0
    return sorted_arr[n//2]

# 策略 2: 自己用循环实现
@njit
def manual_median(arr):
    """手写中位数（Numba 全加速）"""
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    if n % 2 == 0:
        return (sorted_arr[n//2 - 1] + sorted_arr[n//2]) / 2.0
    else:
        return sorted_arr[n//2]
```

---

## 5.8 性能提示：什么时候 Numba 的 NumPy 比原生 NumPy 快

```python
# ❌ 场景：纯 NumPy 向量化操作 → Numba 无法超越
@njit
def add_arrays(a, b):
    return a + b               # 这个 Numba 和 NumPy 一样快，没有加速

# ✅ 场景：循环 + NumPy 逐个元素操作 → Numba 大幅领先
@njit
def complex_loop(arr):
    result = np.zeros_like(arr)
    for i in range(1, len(arr)-1):
        # 三体运算 + 条件判断 —— 这种 NumPy 写起来麻烦且中间数组多
        if arr[i] > 0:
            result[i] = np.sqrt(arr[i-1] + arr[i] + arr[i+1])
        else:
            result[i] = -np.sqrt(abs(arr[i]))
    return result
```

> **规则**：如果你能用一行 `np.sum(arr + b * c / d)` 表达计算，Numba 不会比 NumPy 快。如果你的计算包含不可向量化的循环、分支、迭代，Numba 就是你的加速器。

---

## 5.9 本章关键概念

| 概念 | 一句话 |
|------|--------|
| **数组创建** | `np.zeros/ones/empty` 全部可用，优先用 `np.empty` + 填充来避免重复初始化 |
| **元素运算** | 和 NumPy 完全一致，但 `and/or` 要用 `&/\|` |
| **随机数** | 用 `np.random.*` 语法，但 Numba 有独立的随机数状态 |
| **`np.linalg`** | 大部分支持，但 `svd` 等细节有差异 |
| **不支持列表** | `median`、`percentile`、`histogram`、`unique` 需要自己写 |
| **移除的函数** | `np.trapz` → `np.trapezoid`；`np.in1d` → `np.isin` |
| **混合策略** | Numba 处理循环密集部分，NumPy 处理向量化部分，两者互补 |

---

## 5.10 下章预告

第六章将进入**性能优化策略**——理解了 Numba 能做什么，接下来讲如何让它做得最快：循环优化、内存布局、fastmath、编译缓存等高级技巧。
