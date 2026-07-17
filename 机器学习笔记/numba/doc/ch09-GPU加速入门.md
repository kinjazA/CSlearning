# 第九章：GPU 加速入门

> **核心问题**：CPU 有 8~16 核，GPU 有 2000+ 核。怎么用 Numba 把计算搬到 GPU 上，获得 10~100 倍的加速？

---

## 9.1 GPU 编程基础

### 9.1.1 CPU vs GPU 架构对比

```
CPU (8核):
┌──────────────────────────────────┐
│ Core 1 │ Core 2 │ ... │ Core 8  │  ← 每个核心很强，频率高
│ [复杂] │ [复杂] │     │ [复杂]  │     适合：分支多、逻辑复杂
└──────────────────────────────────┘

GPU (2048核):
┌────────────────────────────────────────────┐
│ C1 C2 ... C32 │ C1 C2 ... C32 │ ... (64组)│  ← 每个核心很简单
│ [简单SIMD]    │ [简单SIMD]    │           │     适合：数据并行
└────────────────────────────────────────────┘
```

| 维度 | CPU | GPU |
|------|-----|-----|
| **核心数** | 8~128 | 1000~10000+ |
| **单核性能** | 强（3~5 GHz） | 弱（1~2 GHz） |
| **擅长** | 复杂逻辑、分支、串行任务 | 大规模数据并行 |
| **内存带宽** | ~50 GB/s | ~1000 GB/s |
| **适用** | 通用计算 | 矩阵运算、图像处理、深度学习 |

### 9.1.2 CUDA 编程模型

CUDA（Compute Unified Device Architecture）是 NVIDIA GPU 的编程接口。Numba 通过 `numba-cuda` 提供 Python 友好的 CUDA 编程。

**核心概念**：

```
主机 (Host) = CPU + 主存
设备 (Device) = GPU + 显存

执行流程：
1. CPU 分配显存
2. CPU 将数据从主存拷贝到显存
3. CPU 启动 GPU 核函数（kernel）
4. GPU 并行执行
5. CPU 将结果从显存拷贝回主存
```

---

## 9.2 安装 `numba-cuda`

### 9.2.1 前提条件

- **硬件**：NVIDIA GPU（支持 CUDA Compute Capability ≥ 3.5）
- **驱动**：NVIDIA 驱动（建议最新版）
- **CUDA Toolkit**：可选（Numba 自带轻量级 CUDA 运行时）

### 9.2.2 安装命令

```bash
# 从 Numba 0.54+ 开始，CUDA 支持分离到独立包
uv pip install numba-cuda
```

### 9.2.3 验证安装

```python
from numba import cuda

# 检查是否检测到 GPU
print(cuda.is_available())        # True/False
print(cuda.gpus)                  # GPU 列表
print(cuda.detect())              # 详细信息

# 输出示例：
# True
# <Managed Device 0>
# Found 1 CUDA devices
# id 0    b'NVIDIA GeForce RTX 3080'
#                               compute capability: 8.6
#                                    pci device id: 0
#                                       pci bus id: 1
```

---

## 9.3 第一个 GPU 程序

### 9.3.1 `@cuda.jit` —— GPU 核函数

```python
from numba import cuda
import numpy as np

@cuda.jit
def add_kernel(a, b, c):
    """GPU 核函数：每个线程计算一个元素"""
    i = cuda.grid(1)              # 获取当前线程的全局索引
    if i < c.size:
        c[i] = a[i] + b[i]

# 主机端代码
n = 1000000
a = np.ones(n, dtype=np.float32)
b = np.ones(n, dtype=np.float32)
c = np.zeros(n, dtype=np.float32)

# 配置：每个 block 有 256 个线程
threads_per_block = 256
blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

# 启动 GPU 核函数
add_kernel[blocks_per_grid, threads_per_block](a, b, c)

print(c[:10])  # [2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]
```

### 9.3.2 核函数的特点

```python
@cuda.jit
def kernel_demo(arr):
    # ✅ 可以：基本运算、NumPy 函数、条件判断
    i = cuda.grid(1)
    if i < len(arr):
        arr[i] = arr[i] * 2 + 1
    
    # ❌ 不可以：return、print、Python 对象
    # return arr[i]          # 核函数无返回值
    # print(arr[i])          # 不能 print
    # lst = [1, 2, 3]        # 不能用 Python list
```

---

## 9.4 CUDA 线程模型

### 9.4.1 Grid / Block / Thread 三层结构

```
Grid (整个任务)
  └─ Block 0
      ├─ Thread 0
      ├─ Thread 1
      └─ ...
  └─ Block 1
      ├─ Thread 0
      └─ ...
  └─ ...

配置示例：
kernel[blocks_per_grid, threads_per_block](...)
       └── Grid 有多少个 Block
                            └── 每个 Block 有多少个 Thread
```

**典型配置**：

```python
# 一维数组：100万个元素
n = 1000000
threads_per_block = 256
blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
# → 3907 个 block，每个 256 个线程

# 二维数组：1024×1024 矩阵
rows, cols = 1024, 1024
threads_per_block = (16, 16)       # 每个 block 是 16×16
blocks_per_grid = ((rows + 15) // 16, (cols + 15) // 16)
# → 64×64 的 block 网格
```

### 9.4.2 获取线程索引

```python
@cuda.jit
def index_demo_1d(arr):
    """一维索引"""
    i = cuda.grid(1)               # 等价于下面
    # i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

@cuda.jit
def index_demo_2d(mat):
    """二维索引"""
    i, j = cuda.grid(2)            # 等价于下面
    # i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    # j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    
    if i < mat.shape[0] and j < mat.shape[1]:
        mat[i, j] = i + j
```

---

## 9.5 内存管理

### 9.5.1 主存 ↔ 显存的数据拷贝

```python
# 方法 1: 自动管理（推荐）
a_host = np.ones(1000000, dtype=np.float32)
a_device = cuda.to_device(a_host)      # CPU → GPU
result_host = a_device.copy_to_host()  # GPU → CPU

# 方法 2: 手动分配
a_device = cuda.device_array(1000000, dtype=np.float32)  # 在 GPU 上分配
cuda.to_device(a_host, to=a_device)    # 拷贝数据到已分配的显存

# 方法 3: 托管内存（自动分页）
a_managed = cuda.managed_array(1000000, dtype=np.float32)
# CPU 和 GPU 都能直接访问，系统自动处理数据迁移
```

### 9.5.2 共享内存（Shared Memory）

每个 block 内的线程可以共享一块高速缓存：

```python
@cuda.jit
def shared_memory_demo(arr, out):
    # 分配共享内存（block 内所有线程共享）
    shared = cuda.shared.array(256, dtype=np.float32)
    
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    i = tx + bx * cuda.blockDim.x
    
    if i < arr.size:
        shared[tx] = arr[i]         # 加载到共享内存
    
    cuda.syncthreads()              # 同步：等所有线程都加载完
    
    if i < arr.size:
        out[i] = shared[tx] * 2     # 从共享内存读取
```

---

## 9.6 性能优化要点

### 9.6.1 最大化并行度

```python
# ❌ 太少线程，GPU 利用率低
threads_per_block = 32
blocks_per_grid = 10
# 总线程数 = 32 × 10 = 320 → GPU 有 2048 核心，大部分在空闲

# ✅ 充分利用 GPU
threads_per_block = 256
blocks_per_grid = 1000
# 总线程数 = 256000 → GPU 满载
```

**经验值**：
- `threads_per_block` 通常是 128 / 256 / 512（2的幂）
- 总线程数 ≥ GPU 核心数 × 2~4

### 9.6.2 合并内存访问（Coalesced Access）

```python
# ✅ 合并访问：同一 warp 的线程访问连续内存
@cuda.jit
def good_access(arr, out):
    i = cuda.grid(1)
    if i < arr.size:
        out[i] = arr[i] * 2     # 线程 0 访问 arr[0]，线程 1 访问 arr[1]...

# ❌ 非合并访问：跳跃访问，带宽浪费
@cuda.jit
def bad_access(arr, out):
    i = cuda.grid(1)
    if i < arr.size:
        out[i] = arr[i * 2]     # 线程访问位置不连续
```

### 9.6.3 避免线程分歧（Thread Divergence）

```python
# ❌ 分歧：同一 warp 的线程走不同分支
@cuda.jit
def divergent(arr):
    i = cuda.grid(1)
    if i % 2 == 0:
        arr[i] = arr[i] * 2     # 偶数线程走这里
    else:
        arr[i] = arr[i] + 1     # 奇数线程走这里
    # → warp 内线程分歧，两个分支串行执行

# ✅ 减少分歧：让连续线程走同一路径
@cuda.jit
def less_divergent(arr):
    i = cuda.grid(1)
    warp_id = i // 32
    if warp_id % 2 == 0:
        arr[i] = arr[i] * 2
    else:
        arr[i] = arr[i] + 1
    # → 同一 warp 的线程走同一分支
```

---

## 9.7 CPU vs GPU 性能对比

```python
import time
from numba import cuda, njit

@njit
def cpu_add(a, b, c):
    for i in range(len(a)):
        c[i] = a[i] + b[i]

@cuda.jit
def gpu_add(a, b, c):
    i = cuda.grid(1)
    if i < c.size:
        c[i] = a[i] + b[i]

n = 10_000_000
a = np.ones(n, dtype=np.float32)
b = np.ones(n, dtype=np.float32)

# CPU 版本
c_cpu = np.zeros(n, dtype=np.float32)
t0 = time.perf_counter()
cpu_add(a, b, c_cpu)
t_cpu = time.perf_counter() - t0

# GPU 版本
a_gpu = cuda.to_device(a)
b_gpu = cuda.to_device(b)
c_gpu = cuda.device_array(n, dtype=np.float32)
threads_per_block = 256
blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

t0 = time.perf_counter()
gpu_add[blocks_per_grid, threads_per_block](a_gpu, b_gpu, c_gpu)
cuda.synchronize()  # 等 GPU 完成
t_gpu = time.perf_counter() - t0

print(f"CPU: {t_cpu:.4f}s")
print(f"GPU: {t_gpu:.4f}s, 加速 {t_cpu/t_gpu:.1f}×")
```

---

## 9.8 GPU 编程的限制

| 限制 | 说明 |
|------|------|
| ❌ 核函数无返回值 | 结果写在参数数组里 |
| ❌ 不能动态分配内存 | 不能在核函数里 `malloc` / `new` |
| ❌ 不能递归 | GPU 不支持递归调用 |
| ❌ 不能用 Python 对象 | 只能用 NumPy 数组、标量 |
| ❌ 不能 `print` | 调试困难（可用 `cuda-gdb`） |
| ⚠️ 数据传输开销 | CPU↔GPU 拷贝可能比计算还慢 |

---

## 9.9 何时用 GPU

```
┌──────────────────────────────────────────┐
│ GPU 加速的判断标准                        │
├──────────────────────────────────────────┤
│                                          │
│ ✅ 应该用 GPU:                            │
│   - 数据量 > 1000万                       │
│   - 高度并行（矩阵运算、卷积）             │
│   - 计算密集（每个元素计算复杂）           │
│   - 数据传输占比 < 20%                    │
│                                          │
│ ❌ 不该用 GPU:                            │
│   - 数据量 < 10万                         │
│   - 逻辑复杂（大量分支）                  │
│   - 数据传输占比 > 50%                    │
│   - 没有 NVIDIA GPU                      │
│                                          │
└──────────────────────────────────────────┘
```

---

## 9.10 本章关键概念

| 概念 | 一句话 |
|------|--------|
| **`@cuda.jit`** | GPU 核函数装饰器，定义在 GPU 上并行执行的代码 |
| **Grid/Block/Thread** | 三层线程组织：Grid 包含多个 Block，Block 包含多个 Thread |
| **`cuda.grid(1)`** | 获取当前线程的全局索引 |
| **显存管理** | `cuda.to_device()` / `cuda.copy_to_host()` / `cuda.device_array()` |
| **共享内存** | Block 内线程共享的高速缓存，`cuda.shared.array()` |
| **合并访问** | 同一 warp 的线程访问连续内存 → 最大化带宽 |
| **线程分歧** | 同一 warp 线程走不同分支 → 串行执行，性能下降 |
| **何时用 GPU** | 数据量大 + 高度并行 + 计算密集 + 传输开销小 |

---

## 9.11 下章预告

第十章：**进阶特性** —— `@jitclass`（编译 Python 类）、`@generated_jit`（元编程）、`@overload`（自定义函数重载）。
