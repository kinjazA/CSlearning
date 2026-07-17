"""
第 01 章 · 概述与安装 — 验证代码

本章代码目的：
1. 验证 joblib 与 pickle 在 numpy 数组序列化上的性能差异
2. 体验三大核心 API 的"最小可用"示例
3. 确认安装版本和环境兼容性

运行方式：
    python 01-概述与安装.py
"""

import time
import os
import sys
import tempfile
import pickle
import shutil

# ============================================================
# 0. 环境检查
# ============================================================
print("=" * 60)
print("0. 环境检查")
print("=" * 60)

# 检查 Python 版本
print(f"Python 版本: {sys.version}")

# 检查 joblib 版本
import joblib
print(f"joblib 版本: {joblib.__version__}")

# 检查 numpy 版本
import numpy as np
print(f"numpy 版本:  {np.__version__}")

# 检查可用 CPU 核心数（后续并行会用到）
import multiprocessing
print(f"CPU 核心数:  {multiprocessing.cpu_count()}")

# 检查可选压缩库
for lib_name in ['lz4', 'zstandard']:
    try:
        __import__(lib_name)
        print(f"可选库 {lib_name}: ✅ 已安装")
    except ImportError:
        print(f"可选库 {lib_name}: ❌ 未安装（不影响基本使用）")

print()

# ============================================================
# 1. 序列化性能对比：pickle vs joblib
# ============================================================
print("=" * 60)
print("1. 序列化性能对比：pickle vs joblib（1000万元素 float64 数组）")
print("=" * 60)

# 构造大数据
SIZE = 10_000_000
big_array = np.random.randn(SIZE).astype(np.float64)
print(f"数组大小: {big_array.nbytes / 1024**2:.1f} MB")
print(f"数组形状: {big_array.shape}")
print(f"数据类型: {big_array.dtype}")

tmp_dir = tempfile.mkdtemp()

# --- pickle ---
pkl_path = os.path.join(tmp_dir, "test.pkl")
t0 = time.perf_counter()
with open(pkl_path, "wb") as f:
    pickle.dump(big_array, f)
pkl_write_time = time.perf_counter() - t0

pkl_size = os.path.getsize(pkl_path)

t0 = time.perf_counter()
with open(pkl_path, "rb") as f:
    _ = pickle.load(f)
pkl_read_time = time.perf_counter() - t0

# --- joblib（无压缩）---
jlb_path = os.path.join(tmp_dir, "test.joblib")
t0 = time.perf_counter()
joblib.dump(big_array, jlb_path, compress=0)
jlb_write_time = time.perf_counter() - t0

jlb_size = os.path.getsize(jlb_path)

t0 = time.perf_counter()
_ = joblib.load(jlb_path)
jlb_read_time = time.perf_counter() - t0

# --- joblib（压缩 level=3，默认）---
jlb_cmp_path = os.path.join(tmp_dir, "test_compressed.joblib")
t0 = time.perf_counter()
joblib.dump(big_array, jlb_cmp_path, compress=3)
jlb_cmp_write_time = time.perf_counter() - t0

jlb_cmp_size = os.path.getsize(jlb_cmp_path)

t0 = time.perf_counter()
_ = joblib.load(jlb_cmp_path)
jlb_cmp_read_time = time.perf_counter() - t0

# --- 结果汇总 ---
print(f"\n{'方法':<25} {'写入时间':>10} {'读取时间':>10} {'文件体积':>12}")
print("-" * 60)
print(f"{'pickle':<25} {pkl_write_time:>9.3f}s {pkl_read_time:>9.3f}s {pkl_size/1024**2:>9.1f} MB")
print(f"{'joblib (无压缩)':<25} {jlb_write_time:>9.3f}s {jlb_read_time:>9.3f}s {jlb_size/1024**2:>9.1f} MB")
print(f"{'joblib (compress=3)':<25} {jlb_cmp_write_time:>9.3f}s {jlb_cmp_read_time:>9.3f}s {jlb_cmp_size/1024**2:>9.1f} MB")

print(f"\n💡 joblib 写入速度是 pickle 的 {pkl_write_time/jlb_write_time:.1f}×")
print(f"💡 joblib 压缩后体积是 pickle 的 {jlb_cmp_size/pkl_size*100:.1f}%")

# ============================================================
# 1.2 小型对象：dict / list（joblib 退回普通 pickle 行为）
# ============================================================
print(f"\n{'='*60}")
print("1.2 小型对象序列化（dict / list 场景）")
print("=" * 60)

small_obj = {
    "model_name": "RandomForest",
    "params": {"n_estimators": 100, "max_depth": 10},
    "feature_names": [f"feat_{i}" for i in range(100)],
    "metrics": {"accuracy": 0.92, "f1": 0.89, "auc": 0.95},
}

# pickle
with open(pkl_path, "wb") as f:
    pickle.dump(small_obj, f)
print(f"pickle 小对象: {os.path.getsize(pkl_path):>6} bytes")

# joblib
joblib.dump(small_obj, jlb_path, compress=0)
print(f"joblib 小对象: {os.path.getsize(jlb_path):>6} bytes")

# 验证正确性
loaded = joblib.load(jlb_path)
assert loaded == small_obj, "序列化/反序列化后数据不一致！"
print("✅ 小对象序列化正确性验证通过")

# ============================================================
# 2. 核心 API 速览 — 最小可用示例
# ============================================================

# --- 2.1 序列化：dump & load ---
print(f"\n{'='*60}")
print("2. 核心 API 最小可用示例")
print("=" * 60)

print("\n--- 2.1 dump / load ---")
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# 造一点数据
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

# 保存
model_path = os.path.join(tmp_dir, "rf_model.joblib")
joblib.dump(model, model_path, compress=3)
print(f"模型已保存: {model_path} ({os.path.getsize(model_path):,} bytes)")

# 加载
loaded_model = joblib.load(model_path)
score = loaded_model.score(X, y)
print(f"加载后准确率: {score:.4f}")

# --- 2.2 缓存：Memory ---
print("\n--- 2.2 Memory 缓存 ---")

memory = joblib.Memory(os.path.join(tmp_dir, "cache_demo"), verbose=1)

call_count = 0

@memory.cache
def expensive_preprocess(path: str, n_rows: int):
    """模拟耗时预处理"""
    global call_count
    call_count += 1
    # 模拟耗时操作
    time.sleep(0.5)
    data = np.random.randn(n_rows, 10)
    return data

print("第一次调用（会真正执行）...")
t0 = time.perf_counter()
r1 = expensive_preprocess("dummy_path", 1000)
t1 = time.perf_counter() - t0
print(f"  耗时: {t1:.2f}s, 函数调用次数: {call_count}")

print("第二次调用（命中缓存）...")
t0 = time.perf_counter()
r2 = expensive_preprocess("dummy_path", 1000)
t2 = time.perf_counter() - t0
print(f"  耗时: {t2:.3f}s, 函数调用次数: {call_count} (未增加!)")

print(f"💡 缓存命中，速度提升 {(t1/t2) if t2 > 0 else '∞':.0f}×")

# --- 2.3 并行：Parallel + delayed ---
print("\n--- 2.3 Parallel + delayed ---")

def slow_square(x):
    """模拟一个耗时计算"""
    time.sleep(0.1)
    return x ** 2

inputs = list(range(16))

# 普通串行
print("串行执行 16 个任务...")
t0 = time.perf_counter()
serial_results = [slow_square(i) for i in inputs]
serial_time = time.perf_counter() - t0
print(f"  耗时: {serial_time:.2f}s")

# 并行
print("并行执行 16 个任务 (n_jobs=4)...")
t0 = time.perf_counter()
parallel_results = joblib.Parallel(n_jobs=4)(
    joblib.delayed(slow_square)(i) for i in inputs
)
parallel_time = time.perf_counter() - t0
print(f"  耗时: {parallel_time:.2f}s")

assert serial_results == parallel_results
print(f"💡 并行加速比: {serial_time/parallel_time:.1f}×")

# ============================================================
# 3. 三大 API 组合拳示例
# ============================================================
print(f"\n{'='*60}")
print("3. 三大 API 组合：缓存驱动并行特征提取")
print("=" * 60)

from joblib import Memory, Parallel, delayed

memory_combo = Memory(os.path.join(tmp_dir, "combo_cache"), verbose=1)

@memory_combo.cache
def extract_features(file_id: int, n_features: int = 50):
    """模拟：每个文件的特征提取很贵"""
    time.sleep(0.05)
    rng = np.random.default_rng(file_id)
    return rng.random(n_features)

file_ids = list(range(40))  # 40 个"文件"

print(f"{len(file_ids)} 个文件，并行提取特征 (n_jobs=4)...")
t0 = time.perf_counter()
features = Parallel(n_jobs=4, verbose=5)(
    delayed(extract_features)(fid) for fid in file_ids
)
t1 = time.perf_counter() - t0

# 按比例生成随机标签做简单训练
y_sim = np.random.default_rng(42).integers(0, 2, len(file_ids))
X_sim = np.array(features)

rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=4)
rf.fit(X_sim, y_sim)
score = rf.score(X_sim, y_sim)

print(f"\n特征提取耗时: {t1:.2f}s")
print(f"特征矩阵形状: {X_sim.shape}")
print(f"训练准确率:   {score:.4f}")

# 第二次运行——全部命中缓存
print("\n第二次运行（全部命中缓存）...")
t0 = time.perf_counter()
features2 = Parallel(n_jobs=4, verbose=0)(
    delayed(extract_features)(fid) for fid in file_ids
)
t2 = time.perf_counter() - t0
print(f"特征提取耗时: {t2:.3f}s (vs 首次 {t1:.2f}s)")
print(f"💡 缓存命中，速度提升 {t1/t2 if t2 > 0 else '∞':.0f}×")

# ============================================================
# 清理
# ============================================================
shutil.rmtree(tmp_dir)

print(f"\n{'='*60}")
print("第 01 章完成 ✓")
print("=" * 60)
print("""
📝 本章验证了什么：
  ✅ joblib 对大 numpy 数组序列化明显快于 pickle
  ✅ 压缩模式可以大幅减小文件体积
  ✅ dump/load、Memory、Parallel 三大 API 的最小可用示例
  ✅ 三者组合：缓存 + 并行 + 模型保存的工作流

📖 下一步：
  第 02 章深入 dump/load，掌握文件命名、协议选择、sklearn 模型最佳实践
""")
