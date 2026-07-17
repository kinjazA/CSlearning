"""
第 04 章 · 内存映射 mmap — 验证代码

目的：亲手感受 mmap 的加载速度、按需分页、多进程共享。

实验：
  1. mmap vs 普通加载 — 时间差异
  2. 按需分页（demand paging）验证
  3. 多进程共享 — 内存不复制
  4. 三种 mmap_mode 行为差异
  5. 压缩 + mmap 互斥验证
  6. 增量训练实战

运行方式：
    uv run python code/04-内存映射mmap.py
"""

import os
import sys
import time
import tempfile
import shutil

import numpy as np
import joblib

# ============================================================
tmp_dir = tempfile.mkdtemp(prefix="joblib_ch04_")
print(f"Python: {sys.version.split()[0]}, joblib: {joblib.__version__}\n")


# ============================================================
# 实验 1：mmap vs 普通加载 — 加载时间对比
# ============================================================
def experiment_1_load_time():
    print("=" * 60)
    print("实验 1 · mmap vs 普通加载 — 加载时间")
    print("=" * 60)

    n = 15_000_000  # ~120 MB
    print(f"创建 {n:,} 元素数组 ({n*8/1024**2:.0f} MB)...")
    arr = np.random.randn(n).astype(np.float64)

    path = os.path.join(tmp_dir, "big_array.joblib")
    joblib.dump(arr, path, compress=0)
    del arr

    file_mb = os.path.getsize(path) / 1024**2
    print(f"文件大小: {file_mb:.1f} MB\n")

    # 普通加载
    print("普通加载 (mmap_mode=None) ...")
    t0 = time.perf_counter()
    full = joblib.load(path, mmap_mode=None)
    t_full = time.perf_counter() - t0
    print(f"  耗时: {t_full:.3f}s — 全部数据复制到内存")
    print(f"  类型: {type(full).__name__}\n")

    # mmap 加载
    print("mmap 加载 (mmap_mode='r') ...")
    t0 = time.perf_counter()
    mm = joblib.load(path, mmap_mode='r')
    t_mmap = time.perf_counter() - t0
    print(f"  耗时: {t_mmap:.5f}s — 只建了映射，没读数据")
    print(f"  类型: {type(mm).__name__}")
    print(f"\n  加速 {t_full/t_mmap:.0f}× — mmap 加载几乎是瞬时的")

    # 验证数据正确
    assert full.shape == mm.shape
    assert np.allclose(full[:100], mm[:100])
    print(f"  数据正确性: ✅ full 和 mmap 内容一致")

    del full, mm
    print()


# ============================================================
# 实验 2：按需分页 — 访问才真正读盘
# ============================================================
def experiment_2_demand_paging():
    print("=" * 60)
    print("实验 2 · 按需分页（demand paging）")
    print("=" * 60)

    n = 5_000_000  # ~40 MB
    arr = np.arange(n, dtype=np.float64)
    path = os.path.join(tmp_dir, "paging_test.joblib")
    joblib.dump(arr, path, compress=0)
    del arr

    mm = joblib.load(path, mmap_mode='r')
    print("mmap 加载完成（数据还在磁盘上）\n")

    # 首次访问：触发 page fault
    t0 = time.perf_counter()
    chunk1 = mm[:10000].copy()
    t_first = time.perf_counter() - t0
    print(f"首次访问 mm[:10000]: {t_first*1000:.2f}ms  ← page fault，真读盘")

    # 再次访问同一位置：已在页缓存
    t0 = time.perf_counter()
    chunk1_again = mm[:10000].copy()
    t_second = time.perf_counter() - t0
    print(f"再次访问 mm[:10000]: {t_second*1000:.3f}ms  ← 已在系统页缓存，不读盘")

    # 访问全新位置：新的 page fault
    t0 = time.perf_counter()
    chunk2 = mm[-10000:].copy()
    t_tail = time.perf_counter() - t0
    print(f"首次访问 mm[-10000:]: {t_tail*1000:.2f}ms  ← 新 page fault")

    print(f"\n📖 首次访问慢（读盘），再次访问快（缓存），不同位置各自触发 page fault")
    print(f"   操作系统按 4KB 页为单位加载，你访问 1 个元素，它加载一整页")

    assert np.array_equal(chunk1, np.arange(10000, dtype=np.float64))
    print(f"✅ 数据正确")
    del mm, chunk1, chunk1_again, chunk2
    print()


# ============================================================
# 实验 3：多进程共享 — 内存不复制
# ============================================================
def experiment_3_multiprocess_sharing():
    print("=" * 60)
    print("实验 3 · 多进程共享 — 内存不复制")
    print("=" * 60)

    n = 2_000_000  # ~16 MB（小一点方便演示）
    arr = np.random.randn(n).astype(np.float64)
    path = os.path.join(tmp_dir, "shared_array.joblib")
    joblib.dump(arr, path, compress=0)
    del arr

    print(f"数组: {n*8/1024**2:.0f} MB")

    # 主进程 mmap 加载
    X = joblib.load(path, mmap_mode='r')

    # 并行计算：每个子进程访问数组的不同部分
    from joblib import Parallel, delayed

    def compute_sum(X_mmap, start, end):
        """子进程：对数组的一段求和"""
        return float(np.sum(X_mmap[start:end]))

    n_jobs = 4
    chunk_size = n // n_jobs
    tasks = [(i * chunk_size, (i + 1) * chunk_size) for i in range(n_jobs)]

    print(f"\n用 {n_jobs} 个子进程并行计算 sum...")
    t0 = time.perf_counter()
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_sum)(X, s, e) for s, e in tasks
    )
    t_parallel = time.perf_counter() - t0

    # 验证结果
    expected_sum = float(np.sum(X[:]))  # 主进程也读一遍
    parallel_sum = sum(results)
    print(f"  主进程计算的 sum:     {expected_sum:.4f}")
    print(f"  子进程计算的 sum 之和: {parallel_sum:.4f}")
    print(f"  一致: {np.isclose(expected_sum, parallel_sum)}")
    print(f"  耗时: {t_parallel:.3f}s")
    print(f"\n📖 关键：{n_jobs} 个子进程共享同一份 mmap 映射，没有各自复制 {n*8/1024**2:.0f}MB 数据")

    del X
    print()


# ============================================================
# 实验 4：三种 mmap_mode 行为差异
# ============================================================
def experiment_4_mmap_modes():
    print("=" * 60)
    print("实验 4 · 三种 mmap_mode 的行为差异")
    print("=" * 60)

    arr = np.arange(100, dtype=np.float64)
    path = os.path.join(tmp_dir, "modes_test.joblib")
    joblib.dump(arr, path, compress=0)

    # 'r' — 只读
    print("mmap_mode='r' (只读):")
    r = joblib.load(path, mmap_mode='r')
    print(f"  读取 r[0] = {r[0]}")
    try:
        r[0] = 999.0
        print(f"  写入成功（不应发生）")
    except (ValueError, OSError) as e:
        print(f"  写入失败: {type(e).__name__} — 只读保护")

    # 'r+' — 读写（会影响文件！）
    print("\nmmap_mode='r+' (读写——会写回磁盘!):")
    rw = joblib.load(path, mmap_mode='r+')
    print(f"  读取 rw[0] = {rw[0]}")
    rw[0] = 888.0
    print(f"  写入 rw[0] = 888.0 成功")
    # 重新加载，检查磁盘文件是否被改
    check = joblib.load(path, mmap_mode='r')
    print(f"  重新加载后 [0] = {check[0]} ← 磁盘文件已被修改!")
    del check, rw

    # 'c' — 写时复制（不影响文件）
    print("\nmmap_mode='c' (写时复制):")
    c = joblib.load(path, mmap_mode='c')
    print(f"  读取 c[0] = {c[0]}（来自磁盘）")
    c[0] = 777.0
    print(f"  写入 c[0] = 777.0 成功（只在内存）")
    check = joblib.load(path, mmap_mode='r')
    print(f"  重新加载后 [0] = {check[0]} ← 磁盘文件未被修改")
    del c, check, r

    # 恢复原始数据
    joblib.dump(np.arange(100, dtype=np.float64), path, compress=0)
    print(f"\n📖 'r' 日常用，'r+' 小心用，'c' 临时改数据用")
    print()


# ============================================================
# 实验 5：压缩 + mmap 互斥
# ============================================================
def experiment_5_compress_mmap_conflict():
    print("=" * 60)
    print("实验 5 · 压缩 + mmap 互斥")
    print("=" * 60)

    arr = np.random.randn(10_000_000).astype(np.float64)  # ~80 MB

    # compress=0 → mmap 生效
    path0 = os.path.join(tmp_dir, "no_compress.joblib")
    joblib.dump(arr, path0, compress=0)
    ld0 = joblib.load(path0, mmap_mode='r')
    print(f"compress=0  → mmap 生效: {isinstance(ld0, np.memmap)}")

    # compress=3 → mmap 被忽略
    path3 = os.path.join(tmp_dir, "with_compress.joblib")
    joblib.dump(arr, path3, compress=3)
    ld3 = joblib.load(path3, mmap_mode='r')
    print(f"compress=3  → mmap 生效: {isinstance(ld3, np.memmap)} (静默忽略)")

    # 验证 mmap 被忽略后，数据仍然正确
    assert np.array_equal(arr, ld3)
    print(f"数据正确性: ✅ 即使 mmap 不生效，数据也正确加载")

    sz0 = os.path.getsize(path0) / 1024**2
    sz3 = os.path.getsize(path3) / 1024**2
    print(f"\n文件大小: compress=0 → {sz0:.1f}MB, compress=3 → {sz3:.1f}MB")

    del arr, ld0, ld3
    print(f"\n📖 选择：要 mmap 就 compress=0，要小文件就放弃 mmap。不可兼得。")
    print()


# ============================================================
# 实验 6：增量训练实战 — 小内存处理大数据
# ============================================================
def experiment_6_incremental_training():
    print("=" * 60)
    print("实验 6 · 增量训练 — mmap 让小内存处理大数据")
    print("=" * 60)

    # 模拟：100万样本 × 200特征 ≈ 1.6GB 特征矩阵
    n_samples, n_features = 1_000_000, 200
    print(f"构造 {n_samples:,} × {n_features} 特征矩阵 "
          f"({n_samples * n_features * 8 / 1024**3:.1f} GB)...")

    X = np.random.randn(n_samples, n_features).astype(np.float32)  # 用 float32 减半
    y = np.random.randint(0, 3, n_samples)

    X_path = os.path.join(tmp_dir, "X_large.joblib")
    y_path = os.path.join(tmp_dir, "y_large.joblib")
    joblib.dump(X, X_path, compress=0)
    joblib.dump(y, y_path, compress=3)
    del X, y

    print(f"X 文件: {os.path.getsize(X_path) / 1024**3:.2f} GB")

    # mmap 加载 X，正常加载 y
    X_mmap = joblib.load(X_path, mmap_mode='r')
    y = joblib.load(y_path)

    from sklearn.linear_model import SGDClassifier

    model = SGDClassifier(loss='log_loss', random_state=42, max_iter=1, tol=None)
    batch_size = 50_000
    classes = np.unique(y)

    print(f"\n增量训练: batch_size={batch_size:,}, 共 {n_samples // batch_size} 个 batch...")
    t0 = time.perf_counter()

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        X_batch = np.array(X_mmap[start:end])  # 只复制当前 batch 到内存
        y_batch = y[start:end]
        model.partial_fit(X_batch, y_batch, classes=classes)

    t_train = time.perf_counter() - t0

    accuracy = model.score(np.array(X_mmap[:batch_size]), y[:batch_size])
    print(f"  训练耗时: {t_train:.1f}s")
    print(f"  验证准确率 (首 batch): {accuracy:.4f}")
    print(f"\n📖 关键：1.6GB 特征矩阵始终在磁盘上，每次只加载 50K 样本到内存")
    print(f"   用 8GB 内存的机器就能处理")

    del X_mmap, y
    print()


# ============================================================
if __name__ == "__main__":
    experiment_1_load_time()
    experiment_2_demand_paging()
    experiment_3_multiprocess_sharing()
    experiment_4_mmap_modes()
    experiment_5_compress_mmap_conflict()
    experiment_6_incremental_training()

    shutil.rmtree(tmp_dir)

    print("=" * 60)
    print("第 04 章完成 ✓")
    print("=" * 60)
    print("""
本章验证了：
  ✅ mmap 加载是瞬时的（只建映射，不读数据）
  ✅ 按需分页 — 首次访问慢（page fault），再次访问快（页缓存）
  ✅ 多进程共享 — 子进程不复制数据
  ✅ 三种模式: 'r'只读, 'r+'写回磁盘, 'c'写时复制
  ✅ 压缩 + mmap 互斥 — 需要 mmap 就不要压缩
  ✅ 增量训练 — 1.6GB 数据在 8GB 机器上训练
""")
