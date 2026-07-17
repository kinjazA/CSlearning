"""
第九章：GPU 加速入门 —— 配套代码
====================================
⚠️ 重要：本章代码需要 NVIDIA GPU + numba-cuda

如果没有 GPU，代码会报错。请参考笔记学习概念。

安装：uv pip install numba-cuda

学习目标：
  1. 用 @cuda.jit 编写 GPU 核函数
  2. 理解 Grid/Block/Thread 线程模型
  3. 管理显存和数据传输
  4. 对比 CPU vs GPU 性能
"""
import numpy as np
import time

try:
    from numba import cuda
    GPU_AVAILABLE = cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️ numba-cuda 未安装，GPU 示例无法运行")
    print("   安装：pip install numba-cuda")

if not GPU_AVAILABLE and 'cuda' in dir():
    print("⚠️ 未检测到 NVIDIA GPU，GPU 示例无法运行")
    print(f"   cuda.is_available() = {cuda.is_available()}")

# ══════════════════════════════════════════════
# 示例 1: 检测 GPU
# ══════════════════════════════════════════════

def demo_detect_gpu():
    print("=" * 55)
    print("示例 1: 检测 GPU")
    print("=" * 55)

    if not GPU_AVAILABLE:
        print("  ❌ GPU 不可用")
        print("     需要：NVIDIA GPU + CUDA 驱动 + numba-cuda")
        return False

    print(f"  ✅ GPU 可用")
    print(f"  检测到的 GPU:")
    for gpu in cuda.gpus:
        print(f"    - {gpu.name.decode('utf-8')}")
        print(f"      计算能力: {gpu.compute_capability}")
        print(f"      显存: {gpu.get_memory_info()[1] / 1024**3:.1f} GB")
    return True


# ══════════════════════════════════════════════
# 示例 2: 第一个 GPU 核函数 —— 向量加法
# ══════════════════════════════════════════════

if GPU_AVAILABLE:
    @cuda.jit
    def add_kernel(a, b, c):
        """GPU 核函数：每个线程计算一个元素"""
        i = cuda.grid(1)              # 获取全局线程索引
        if i < c.size:
            c[i] = a[i] + b[i]


def demo_first_kernel():
    print(f"\n{'='*55}")
    print("示例 2: 第一个 GPU 核函数")
    print(f"{'='*55}")

    if not GPU_AVAILABLE:
        print("  ⊘ GPU 不可用，跳过")
        return

    n = 1000000
    a = np.ones(n, dtype=np.float32)
    b = np.ones(n, dtype=np.float32) * 2
    c = np.zeros(n, dtype=np.float32)

    # 配置线程
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

    print(f"  数组大小: {n:,}")
    print(f"  Grid 配置: {blocks_per_grid} blocks × {threads_per_block} threads")
    print(f"  总线程数: {blocks_per_grid * threads_per_block:,}")

    # 启动核函数
    add_kernel[blocks_per_grid, threads_per_block](a, b, c)
    cuda.synchronize()

    print(f"  结果 (前10个): {c[:10]}")
    print(f"  验证: 全部为3? {np.all(c == 3.0)}")


# ══════════════════════════════════════════════
# 示例 3: 显存管理 —— CPU ↔ GPU 数据传输
# ══════════════════════════════════════════════

def demo_memory_transfer():
    print(f"\n{'='*55}")
    print("示例 3: 显存管理")
    print(f"{'='*55}")

    if not GPU_AVAILABLE:
        print("  ⊘ GPU 不可用，跳过")
        return

    n = 10_000_000
    a_host = np.random.randn(n).astype(np.float32)

    # 测量传输时间
    t0 = time.perf_counter()
    a_device = cuda.to_device(a_host)      # CPU → GPU
    t_to_device = time.perf_counter() - t0

    t0 = time.perf_counter()
    a_back = a_device.copy_to_host()       # GPU → CPU
    t_to_host = time.perf_counter() - t0

    data_size_mb = n * 4 / 1024 / 1024
    print(f"  数据大小: {data_size_mb:.1f} MB")
    print(f"  CPU → GPU: {t_to_device*1000:.2f} ms ({data_size_mb/t_to_device:.0f} MB/s)")
    print(f"  GPU → CPU: {t_to_host*1000:.2f} ms ({data_size_mb/t_to_host:.0f} MB/s)")
    print(f"  验证: {np.allclose(a_host, a_back)}")


# ══════════════════════════════════════════════
# 示例 4: CPU vs GPU 性能对比
# ══════════════════════════════════════════════

if GPU_AVAILABLE:
    @cuda.jit
    def gpu_compute_intensive(arr, out):
        """GPU 版本：计算密集型操作"""
        i = cuda.grid(1)
        if i < arr.size:
            x = arr[i]
            # 模拟复杂计算
            for _ in range(100):
                x = x * 0.99 + 0.01
            out[i] = x


from numba import njit

@njit
def cpu_compute_intensive(arr, out):
    """CPU 版本"""
    for i in range(len(arr)):
        x = arr[i]
        for _ in range(100):
            x = x * 0.99 + 0.01
        out[i] = x


def demo_cpu_vs_gpu():
    print(f"\n{'='*55}")
    print("示例 4: CPU vs GPU 性能对比")
    print(f"{'='*55}")

    if not GPU_AVAILABLE:
        print("  ⊘ GPU 不可用，只运行 CPU 版本")
        n = 100000
        arr = np.random.randn(n).astype(np.float32)
        out = np.zeros_like(arr)

        # 热身
        cpu_compute_intensive(arr[:100], out[:100])

        t0 = time.perf_counter()
        cpu_compute_intensive(arr, out)
        t_cpu = time.perf_counter() - t0
        print(f"  CPU (n={n:,}): {t_cpu:.4f}s")
        return

    n = 1_000_000
    arr = np.random.randn(n).astype(np.float32)

    # CPU 版本
    out_cpu = np.zeros_like(arr)
    cpu_compute_intensive(arr[:100], out_cpu[:100])  # 热身

    t0 = time.perf_counter()
    cpu_compute_intensive(arr, out_cpu)
    t_cpu = time.perf_counter() - t0

    # GPU 版本（含传输时间）
    out_gpu = np.zeros_like(arr)
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

    t0 = time.perf_counter()
    arr_d = cuda.to_device(arr)
    out_d = cuda.device_array_like(arr)
    gpu_compute_intensive[blocks_per_grid, threads_per_block](arr_d, out_d)
    out_d.copy_to_host(out_gpu)
    t_gpu_total = time.perf_counter() - t0

    # GPU 版本（不含传输时间）
    arr_d = cuda.to_device(arr)
    out_d = cuda.device_array_like(arr)
    cuda.synchronize()

    t0 = time.perf_counter()
    gpu_compute_intensive[blocks_per_grid, threads_per_block](arr_d, out_d)
    cuda.synchronize()
    t_gpu_compute = time.perf_counter() - t0

    print(f"  数组大小: {n:,}")
    print(f"  CPU:              {t_cpu:.4f}s")
    print(f"  GPU (含传输):     {t_gpu_total:.4f}s  加速 {t_cpu/t_gpu_total:.1f}×")
    print(f"  GPU (纯计算):     {t_gpu_compute:.4f}s  加速 {t_cpu/t_gpu_compute:.1f}×")
    print(f"  传输开销占比:     {(t_gpu_total-t_gpu_compute)/t_gpu_total*100:.1f}%")
    print(f"  结果一致: {np.allclose(out_cpu, out_gpu, rtol=1e-4)}")


# ══════════════════════════════════════════════
# 示例 5: 二维核函数 —— 矩阵加法
# ══════════════════════════════════════════════

if GPU_AVAILABLE:
    @cuda.jit
    def matrix_add_2d(A, B, C):
        """二维核函数"""
        i, j = cuda.grid(2)
        if i < C.shape[0] and j < C.shape[1]:
            C[i, j] = A[i, j] + B[i, j]


def demo_2d_kernel():
    print(f"\n{'='*55}")
    print("示例 5: 二维核函数 —— 矩阵加法")
    print(f"{'='*55}")

    if not GPU_AVAILABLE:
        print("  ⊘ GPU 不可用，跳过")
        return

    rows, cols = 1024, 1024
    A = np.random.randn(rows, cols).astype(np.float32)
    B = np.random.randn(rows, cols).astype(np.float32)
    C = np.zeros((rows, cols), dtype=np.float32)

    # 二维 Grid 配置
    threads_per_block = (16, 16)
    blocks_per_grid_x = (rows + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (cols + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    print(f"  矩阵大小: {rows}×{cols}")
    print(f"  Blocks: {blocks_per_grid}")
    print(f"  Threads per block: {threads_per_block}")

    A_d = cuda.to_device(A)
    B_d = cuda.to_device(B)
    C_d = cuda.device_array_like(C)

    matrix_add_2d[blocks_per_grid, threads_per_block](A_d, B_d, C_d)
    C_d.copy_to_host(C)

    # 验证
    C_numpy = A + B
    print(f"  结果一致: {np.allclose(C, C_numpy)}")


# ══════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("╔═══════════════════════════════════════════╗")
    print("║    Numba 第九章：GPU 加速入门             ║")
    print("║    配套代码演示                          ║")
    print("╚═══════════════════════════════════════════╝")

    has_gpu = demo_detect_gpu()

    if has_gpu:
        demo_first_kernel()
        demo_memory_transfer()
        demo_cpu_vs_gpu()
        demo_2d_kernel()
    else:
        print()
        print("  💡 没有 GPU？你仍然可以：")
        print("     1. 阅读笔记理解 GPU 编程概念")
        print("     2. 在云平台（Google Colab / Kaggle）上运行")
        print("     3. 使用 CPU 并行（第八章）已经能覆盖大部分场景")

    print(f"\n{'='*55}")
    if has_gpu:
        print("✅ 第九章代码演示完成！")
    else:
        print("⊘ 第九章需要 GPU 硬件")
    print(f"{'='*55}")
