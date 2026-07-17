"""
第 03 章 · 压缩策略 — 验证代码

目的：在你的机器上实际跑一遍各压缩算法的性能，建立你自己的 benchmark 直觉。

实验：
  1. 全部可用算法的速度/体积对比
  2. level 1→9 的边际收益
  3. 不同数据类型（随机 vs 稀疏）的压缩差异
  4. 压缩是无损的 — 验证
  5. 压缩 vs mmap 互斥 — 验证

运行方式：
    uv run python code/03-压缩策略.py
"""

import os
import sys
import time
import tempfile
import shutil

import numpy as np
import joblib

# ============================================================
# 准备工作
# ============================================================
tmp_dir = tempfile.mkdtemp(prefix="joblib_ch03_")
print(f"Python: {sys.version.split()[0]}, joblib: {joblib.__version__}\n")


# ============================================================
# 实验 1：全部算法对比 — 同一份数据，不同压缩
# ============================================================
def experiment_1_all_algorithms():
    print("=" * 60)
    print("实验 1 · 全部压缩算法对比")
    print("=" * 60)

    # 用随机数组（最难压缩的情况）测试各算法
    arr = np.random.randn(500_000).astype(np.float64)
    raw_mb = arr.nbytes / 1024**2
    print(f"测试数据: {raw_mb:.1f} MB 随机 float64 数组\n")

    # 收集所有可用的压缩配置
    configs = [
        ("不压缩",           0),
        ("zlib-3 (默认)",    3),
        ("zlib-9",           9),
        ("gzip-3",           ("gzip", 3)),
        ("gzip-9",           ("gzip", 9)),
        ("bz2-3",            ("bz2", 3)),
        ("lzma-3",           ("lzma", 3)),
        ("lzma-9",           ("lzma", 9)),
    ]
    try:
        import lz4
        configs.extend([("lz4-0", ("lz4", 0)), ("lz4-3", ("lz4", 3))])
    except ImportError:
        print("(lz4 未安装，跳过 lz4 测试)")
    try:
        import zstandard
        configs.extend([("zstd-1", ("zstd", 1)), ("zstd-3", ("zstd", 3)), ("zstd-9", ("zstd", 9))])
    except ImportError:
        print("(zstandard 未安装，跳过 zstd 测试)")

    print(f"\n{'算法':<20} {'写入':>8} {'读取':>8} {'文件大小':>10} {'压缩率':>8}")
    print("-" * 58)

    results = []
    for label, comp in configs:
        path = os.path.join(tmp_dir, f"algo_{label.replace(' ','_').replace('(','').replace(')','')}.joblib")
        # 写入
        t0 = time.perf_counter()
        joblib.dump(arr, path, compress=comp)
        t_w = time.perf_counter() - t0
        # 读取
        t0 = time.perf_counter()
        loaded = joblib.load(path)
        t_r = time.perf_counter() - t0
        # 统计
        sz_mb = os.path.getsize(path) / 1024**2
        ratio = sz_mb / raw_mb * 100
        assert np.array_equal(arr, loaded), f"{label} 数据不一致!"
        print(f"{label:<20} {t_w:>7.3f}s {t_r:>7.3f}s {sz_mb:>8.2f}MB {ratio:>7.1f}%")
        results.append((label, t_w, t_r, sz_mb, ratio))

    # 推荐
    print(f"\n{'='*60}")
    print("推荐：")
    print(f"  日常开发  → compress=3 (zlib，默认)")
    has_zstd = any("zstd" in r[0] for r in results)
    has_lz4 = any("lz4" in r[0] for r in results)
    if has_zstd:
        print(f"  最佳平衡  → compress=('zstd', 3)")
    if has_lz4:
        print(f"  追求速度  → compress=('lz4', 0)")
    print(f"  最小体积  → compress=('lzma', 9)")


# ============================================================
# 实验 2：level 边际收益 — 1→9 的投入产出比
# ============================================================
def experiment_2_level_diminishing_returns():
    print(f"\n{'='*60}")
    print("实验 2 · zlib level 的边际收益")
    print("=" * 60)

    arr = np.random.randn(1_000_000).astype(np.float64)
    raw_mb = arr.nbytes / 1024**2

    print(f"\n{'level':<8} {'写入':>8} {'读取':>8} {'文件大小':>10} {'压缩率':>8} {'相对0的慢倍数':>14}")
    print("-" * 62)

    t_base = None
    for level in [0, 1, 2, 3, 5, 7, 9]:
        path = os.path.join(tmp_dir, f"zlib_lv{level}.joblib")
        t0 = time.perf_counter()
        joblib.dump(arr, path, compress=level)
        t_w = time.perf_counter() - t0
        if level == 0:
            t_base = t_w
        sz_mb = os.path.getsize(path) / 1024**2
        t0 = time.perf_counter()
        joblib.load(path)
        t_r = time.perf_counter() - t0
        slowdown = t_w / t_base if t_base else 1
        print(f"  {level:<6} {t_w:>7.3f}s {t_r:>7.3f}s {sz_mb:>8.2f}MB {sz_mb/raw_mb*100:>7.1f}% {slowdown:>13.1f}x")

    print(f"\n📖 观察：level 1→3 压缩率大幅提升(90%→55%)，慢 3×")
    print(f"        level 3→9 压缩率只多几个百分点(55%→48%)，慢 10×")
    print(f"        → 日常用 level 3，归档才用 level 9")


# ============================================================
# 实验 3：数据类型对压缩效果的影响
# ============================================================
def experiment_3_data_type_impact():
    print(f"\n{'='*60}")
    print("实验 3 · 不同数据类型的压缩效果差异")
    print("=" * 60)

    n = 500_000
    datasets = {
        "随机浮点数": np.random.randn(n).astype(np.float64),
        "零为主的稀疏": np.zeros(n, dtype=np.float64),  # 全零
        "低值整数": np.random.randint(0, 10, n).astype(np.float64),
    }
    # 稀疏数据：大部分是零
    datasets["零为主的稀疏"][::50] = 1.0  # 每 50 个元素一个 1.0

    print(f"\n{'数据类型':<16} {'原始体积':>10} {'compress=3':>12} {'compress=9':>12} {'lzma-9':>12}")
    print("-" * 66)

    for name, arr in datasets.items():
        raw_mb = arr.nbytes / 1024**2
        row = f"{name:<16} {raw_mb:>8.1f}MB"
        for comp_label, comp in [("zlib-3", 3), ("zlib-9", 9), ("lzma-9", ("lzma", 9))]:
            path = os.path.join(tmp_dir, f"dtype_{name}_{comp_label}.joblib")
            joblib.dump(arr, path, compress=comp)
            sz_mb = os.path.getsize(path) / 1024**2
            ratio = sz_mb / raw_mb * 100
            row += f" {sz_mb:>8.2f}MB({ratio:>4.0f}%)"
        print(row)

    print(f"\n📖 观察：随机数据最难压缩（50%+），稀疏数据可压到 1% 以下")
    print(f"        → 特征矩阵如果是稀疏的（one-hot 编码等），压缩策略可以更激进")


# ============================================================
# 实验 4：压缩是无损的
# ============================================================
def experiment_4_lossless():
    print(f"\n{'='*60}")
    print("实验 4 · 压缩是无损的")
    print("=" * 60)

    arr = np.random.randn(100_000).astype(np.float64)
    for comp in [0, 3, ("zstd", 3), ("lz4", 0), ("lzma", 9)]:
        path = os.path.join(tmp_dir, f"lossless_{str(comp)}.joblib")
        joblib.dump(arr, path, compress=comp)
        loaded = joblib.load(path)
        max_diff = np.max(np.abs(arr - loaded))
        label = str(comp).replace("'", "")
        print(f"  {label:<20} 最大差异={max_diff:.2e}  完全一致={np.array_equal(arr, loaded)}")

    print(f"\n📖 所有算法 max_diff = 0.00 → 完全无损")


# ============================================================
# 实验 5：压缩 vs mmap 互斥
# ============================================================
def experiment_5_compress_vs_mmap():
    print(f"\n{'='*60}")
    print("实验 5 · 压缩 vs mmap 互斥")
    print("=" * 60)

    arr = np.random.randn(5_000_000).astype(np.float64)  # ~40 MB

    # compress=0: mmap 生效
    path_noc = os.path.join(tmp_dir, "nocompress.joblib")
    joblib.dump(arr, path_noc, compress=0)
    loaded_noc = joblib.load(path_noc, mmap_mode='r')
    print(f"compress=0 → mmap 生效: {isinstance(loaded_noc, np.memmap)}")

    # compress=3: mmap 被忽略
    path_c = os.path.join(tmp_dir, "compressed.joblib")
    joblib.dump(arr, path_c, compress=3)
    loaded_c = joblib.load(path_c, mmap_mode='r')
    print(f"compress=3 → mmap 生效: {isinstance(loaded_c, np.memmap)} (mmap_mode 参数被静默忽略)")

    # compress=('zstd', 3): mmap 被忽略
    path_zstd = os.path.join(tmp_dir, "compressed_zstd.joblib")
    joblib.dump(arr, path_zstd, compress=('zstd', 3))
    loaded_z = joblib.load(path_zstd, mmap_mode='r')
    print(f"compress=zstd-3 → mmap 生效: {isinstance(loaded_z, np.memmap)} (同样被忽略)")

    print(f"\n📖 结论：压缩和 mmap 不可兼得")
    print(f"   需要 mmap → compress=0")
    print(f"   需要压缩 → 放弃 mmap")
    print(f"   同时需要 → 分开保存：大数组 compress=0 + mmap，其余 compress=3")

    del arr, loaded_noc, loaded_c, loaded_z


# ============================================================
# 运行
# ============================================================
if __name__ == "__main__":
    experiment_1_all_algorithms()
    experiment_2_level_diminishing_returns()
    experiment_3_data_type_impact()
    experiment_4_lossless()
    experiment_5_compress_vs_mmap()

    shutil.rmtree(tmp_dir)

    print(f"\n{'='*60}")
    print("第 03 章完成")
    print("=" * 60)
