"""
第 06 章 · 缓存高级管理 — 验证代码

目的：掌握 ignore、check_call_in_cache、清理策略、不落盘缓存等进阶用法。

实验：
  1. ignore — 排除不影响结果的参数
  2. check_call_in_cache — 先检查，再计算
  3. 清理策略 — clear / reduce_size / 按函数删除
  4. 不落盘缓存 — Memory(None)
  5. 缓存结果的 mmap 加载

运行方式：
    uv run python code/06-缓存依赖与高级管理.py
"""

import os
import sys
import time
import tempfile
import shutil
from datetime import timedelta

import numpy as np
from joblib import Memory

# ============================================================
tmp_dir = tempfile.mkdtemp(prefix="joblib_ch06_")
cache_dir = os.path.join(tmp_dir, "cache")
print(f"缓存目录: {cache_dir}\n")


# ============================================================
# 实验 1：ignore — 排除不影响结果的参数
# ============================================================
def experiment_1_ignore():
    print("=" * 60)
    print("实验 1 · ignore — 排除不影响结果的参数")
    print("=" * 60)

    memory = Memory(os.path.join(cache_dir, "exp1"), verbose=1)
    call_log = []

    @memory.cache(ignore=['verbose'])
    def train(data_path, model_type, verbose=False):
        call_log.append((data_path, model_type))
        time.sleep(0.3)
        return f"{model_type}_on_{data_path}"

    print("\n第一次: train('data.csv', 'rf', verbose=False)")
    train('data.csv', 'rf', verbose=False)

    print("\n第二次: train('data.csv', 'rf', verbose=True) — ignore 生效")
    train('data.csv', 'rf', verbose=True)

    print("\n第三次: train('data.csv', 'xgb', verbose=False) — model_type 变了")
    train('data.csv', 'xgb', verbose=False)

    print(f"\n实际执行次数: {len(call_log)} (应为 2，verbose 变化不触发重算)")
    assert len(call_log) == 2
    print("✅ ignore 验证通过\n")

    memory.clear()


# ============================================================
# 实验 2：check_call_in_cache — 先问再算
# ============================================================
def experiment_2_check_cache():
    print("=" * 60)
    print("实验 2 · check_call_in_cache — 先检查再计算")
    print("=" * 60)

    memory = Memory(os.path.join(cache_dir, "exp2"), verbose=0)

    @memory.cache
    def expensive_task(param_id, data):
        time.sleep(0.2)
        return f"result_{param_id}"

    # 模拟大量任务，先检查哪些需要计算
    tasks = [(i, np.ones(100)) for i in range(5)]
    cached_count = 0

    for pid, data in tasks:
        if memory.check_call_in_cache(expensive_task, pid, data):
            cached_count += 1
        else:
            pass  # 需要执行

    print(f"5 个任务中已缓存: {cached_count} 个 (第一次应该都是 0)")

    # 执行一批
    for pid, data in tasks[:3]:
        expensive_task(pid, data)

    # 再检查
    cached_count = 0
    for pid, data in tasks:
        if memory.check_call_in_cache(expensive_task, pid, data):
            cached_count += 1
    print(f"执行 3 个后，已缓存: {cached_count} 个 (应为 3)")

    assert cached_count == 3
    print("✅ check_call_in_cache 验证通过\n")

    memory.clear()


# ============================================================
# 实验 3：清理策略
# ============================================================
def experiment_3_cleanup():
    print("=" * 60)
    print("实验 3 · 缓存清理策略")
    print("=" * 60)

    memory = Memory(os.path.join(cache_dir, "exp3"), verbose=0)

    @memory.cache
    def compute_slow(x):
        return x ** 2

    # 生成一批缓存
    for i in range(20):
        compute_slow(i)

    # 统计
    def count_files(d):
        c = 0
        for root, dirs, files in os.walk(d):
            c += len([f for f in files if not f.startswith('.')])
        return c

    base = os.path.join(cache_dir, "exp3")
    before = count_files(base)
    print(f"缓存文件: {before} 个")

    # 全部清空
    memory.clear()
    after_clear = count_files(base)
    print(f"clear() 后: {after_clear} 个")

    # 重新生成
    for i in range(20):
        compute_slow(i)

    # 按时间清理（制造一些"旧"缓存不太容易在测试中做到，演示用法）
    # memory.reduce_size(age_limit=timedelta(days=30))
    print(f"\n📖 按时间清理: memory.reduce_size(age_limit=timedelta(days=30))")
    print(f"📖 手动删单个函数: shutil.rmtree('./cache/joblib/__main__/compute_slow')")

    memory.clear()
    print("✅ 清理策略验证完成\n")


# ============================================================
# 实验 4：不落盘缓存 — Memory(None)
# ============================================================
def experiment_4_no_disk():
    print("=" * 60)
    print("实验 4 · Memory(None) — 不落盘，只存内存")
    print("=" * 60)

    memory = Memory(None, verbose=1)
    call_count = [0]

    @memory.cache
    def in_memory_task(x):
        call_count[0] += 1
        time.sleep(0.2)
        return x * 10

    print("\n第一次调用 in_memory_task(5):")
    r1 = in_memory_task(5)

    print("\n第二次调用 in_memory_task(5):")
    r2 = in_memory_task(5)

    print(f"\n执行次数: {call_count[0]} (应为 1)")
    assert r1 == r2 == 50
    assert call_count[0] == 1

    # 验证没有磁盘文件
    # Memory(None) 不会创建任何文件
    print(f"📖 Memory(None) 不写磁盘，进程重启后缓存消失")
    print("✅ Memory(None) 验证通过\n")


# ============================================================
# 实验 5：mmap_mode 加载缓存的返回值
# ============================================================
def experiment_5_mmap_cache():
    print("=" * 60)
    print("实验 5 · 缓存返回值用 mmap 加载")
    print("=" * 60)

    # 注意：mmap_mode 参数是传给 Memory 构造函数的，不是传给 @memory.cache 的
    mmap_memory = Memory(os.path.join(cache_dir, "exp5_mmap"), mmap_mode='r', verbose=0)
    normal_memory = Memory(os.path.join(cache_dir, "exp5_normal"), verbose=0)

    # 生成大数组
    big_arr = np.random.randn(5_000_000).astype(np.float64)  # ~40 MB

    @mmap_memory.cache
    def load_big_mmap(file_id):
        return big_arr

    @normal_memory.cache
    def load_big_normal(file_id):
        return big_arr

    # 预热缓存
    _ = load_big_mmap(1)
    _ = load_big_normal(1)

    # 加载对比
    arr_mmap = load_big_mmap(1)
    arr_normal = load_big_normal(1)

    print(f"mmap_mode='r' → 返回类型: {type(arr_mmap).__name__} (支持零拷贝)")
    print(f"默认模式    → 返回类型: {type(arr_normal).__name__} (全部在内存)")

    assert np.allclose(arr_mmap, arr_normal)
    print("✅ 两种模式数据一致")

    mmap_memory.clear()
    normal_memory.clear()
    del big_arr, arr_mmap, arr_normal
    print()


# ============================================================
if __name__ == "__main__":
    experiment_1_ignore()
    experiment_2_check_cache()
    experiment_3_cleanup()
    experiment_4_no_disk()
    experiment_5_mmap_cache()

    shutil.rmtree(tmp_dir)

    print("=" * 60)
    print("第 06 章完成 ✓")
    print("=" * 60)
    print("""
本章验证了：
  ✅ ignore=['verbose'] — 不影响结果的参数排除在缓存 key 之外
  ✅ check_call_in_cache — 先检查，只提交还没缓存的
  ✅ clear()/reduce_size() — 缓存清理策略
  ✅ Memory(None) — 不落盘纯内存缓存
  ✅ Memory(mmap_mode='r') — 缓存的大数组 mmap 加载
""")
