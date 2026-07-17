"""
第 10 章 · ML/DL 综合实战 — 验证代码

目的：用一个完整的 ML 项目，串起 dump/load、Memory、Parallel、mmap、compress。

场景：
  A. 文本分类 — Memory 缓存 + Parallel 调参 + dump 持久化
  B. 图像特征 — mmap + Parallel 批量提取 + 压缩存储
  C. 增量训练 — mmap 大特征矩阵 + SGD 增量训练

运行方式：
    uv run python code/10-ML-DL综合实战.py
"""

import os
import sys
import time
import tempfile
import shutil

import numpy as np
import joblib
from joblib import Memory, Parallel, delayed
from datetime import date

# ============================================================
tmp_dir = tempfile.mkdtemp(prefix="joblib_ch10_")
cache_dir = os.path.join(tmp_dir, "cache")
model_dir = os.path.join(tmp_dir, "models")
os.makedirs(model_dir, exist_ok=True)

TODAY = date.today().isoformat()
print(f"日期: {TODAY}, Python: {sys.version.split()[0]}, CPU: {os.cpu_count()} 核\n")


# ╔══════════════════════════════════════════════════════════════╗
# ║  场景 A · 文本分类 Pipeline                                  ║
# ║  Memory 缓存 + Parallel 调参 + dump 持久化                    ║
# ╚══════════════════════════════════════════════════════════════╝

def scenario_a_text_classification():
    print("=" * 60)
    print("场景 A · 文本分类 — 缓存 + 并行 + 持久化")
    print("=" * 60)

    from sklearn.datasets import make_classification
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import cross_val_score

    memory = Memory(os.path.join(cache_dir, "scenario_a"), verbose=1)

    # 步骤 1：缓存数据生成
    @memory.cache
    def generate_data(n_samples, n_features, seed):
        """生成分类数据——同样参数只跑一次"""
        X, y = make_classification(
            n_samples=n_samples, n_features=n_features,
            n_informative=50, random_state=seed
        )
        return X, y

    print("\n步骤 1: 生成数据（缓存）...")
    X, y = generate_data(5000, 100, seed=42)
    print(f"  数据: X={X.shape}, y={y.shape}")

    # 步骤 2：并行交叉验证（验证单个模型）
    def evaluate_fold(alpha, X, y):
        """单次评估——内层 n_jobs=1"""
        model = SGDClassifier(
            loss='log_loss', alpha=alpha, max_iter=1000,
            n_jobs=1, random_state=42
        )
        scores = cross_val_score(model, X, y, cv=3)
        return {'alpha': alpha, 'mean': float(scores.mean()), 'std': float(scores.std())}

    # 步骤 3：并行超参搜索
    alphas = [0.0001, 0.001, 0.01, 0.1, 1.0]

    print(f"\n步骤 3: 并行搜索 {len(alphas)} 个 alpha 值...")
    t0 = time.perf_counter()
    search_results = Parallel(n_jobs=-1, verbose=10)(
        delayed(evaluate_fold)(a, X, y) for a in alphas
    )
    t_search = time.perf_counter() - t0

    # 找最佳
    best = max(search_results, key=lambda r: r['mean'])
    print(f"\n  最佳 alpha: {best['alpha']}, CV: {best['mean']:.4f} ± {best['std']:.4f}")
    print(f"  搜索耗时: {t_search:.1f}s")

    # 步骤 4：持久化最佳模型
    print(f"\n步骤 4: 训练最终模型并保存...")
    final_model = SGDClassifier(
        loss='log_loss', alpha=best['alpha'], max_iter=1000, random_state=42
    )
    final_model.fit(X, y)

    model_path = os.path.join(model_dir, f"sgd_alpha{best['alpha']}_{TODAY}.joblib")
    joblib.dump(final_model, model_path, compress=3)
    print(f"  模型: {model_path} ({os.path.getsize(model_path) / 1024:.1f} KB)")

    # 步骤 5：验证 roundtrip
    loaded = joblib.load(model_path)
    score_original = final_model.score(X, y)
    score_loaded = loaded.score(X, y)
    print(f"  Roundtrip: 原始={score_original:.4f}, 加载后={score_loaded:.4f}, 一致={score_original==score_loaded}")

    # 步骤 6：第二次调用 → 全部缓存命中
    print(f"\n步骤 6: 第二次生成数据（缓存命中）...")
    X2, y2 = generate_data(5000, 100, seed=42)
    print(f"  数据形状: {X2.shape}")

    memory.clear()
    print(f"\n✅ 场景 A 完成\n")
    return search_results


# ╔══════════════════════════════════════════════════════════════╗
# ║  场景 B · 图像特征提取                                       ║
# ║  mmap 大数组 + Parallel 批量提取 + 压缩存储                   ║
# ╚══════════════════════════════════════════════════════════════╝

def scenario_b_image_features():
    print("=" * 60)
    print("场景 B · 图像特征 — mmap + 并行提取 + 压缩")
    print("=" * 60)

    # 模拟：1000 张"图片"
    def extract_features(image_id):
        """对单张图片提取特征（模拟）"""
        rng = np.random.RandomState(image_id)
        time.sleep(0.005)  # 模拟 IO
        return rng.randn(512).astype(np.float32)  # 512 维特征向量

    n_images = 200

    # 步骤 1：并行提取特征
    print(f"\n步骤 1: 并行提取 {n_images} 张图片的特征...")
    t0 = time.perf_counter()
    features = Parallel(n_jobs=-1, verbose=10)(
        delayed(extract_features)(i) for i in range(n_images)
    )
    t_extract = time.perf_counter() - t0

    X = np.array(features)
    print(f"  特征矩阵: {X.shape}, 耗时: {t_extract:.1f}s")

    # 步骤 2：用压缩存储
    feat_path_c = os.path.join(tmp_dir, f"features_compressed_{TODAY}.joblib")
    joblib.dump(X, feat_path_c, compress=('zstd', 3))
    print(f"\n步骤 2: 压缩存储: {os.path.getsize(feat_path_c)/1024:.1f} KB")

    # 步骤 3：如果是超大规模 → 不压缩 + mmap
    feat_path_nc = os.path.join(tmp_dir, f"features_nocompress_{TODAY}.joblib")
    joblib.dump(X, feat_path_nc, compress=0)
    X_mmap = joblib.load(feat_path_nc, mmap_mode='r')
    print(f"步骤 3: 不压缩 + mmap: {os.path.getsize(feat_path_nc)/1024:.1f} KB")
    print(f"  mmap 加载类型: {type(X_mmap).__name__}")
    print(f"  文件大小比: 压缩={os.path.getsize(feat_path_c)/os.path.getsize(feat_path_nc)*100:.0f}%")

    del X, X_mmap, features
    print(f"\n✅ 场景 B 完成\n")


# ╔══════════════════════════════════════════════════════════════╗
# ║  场景 C · 增量训练（大特征矩阵 + mmap）                       ║
# ║  模拟：特征矩阵太大装不进内存，用 mmap + 增量训练              ║
# ╚══════════════════════════════════════════════════════════════╝

def scenario_c_incremental_training():
    print("=" * 60)
    print("场景 C · 增量训练 — mmap 大矩阵 + SGD")
    print("=" * 60)

    from sklearn.linear_model import SGDClassifier

    # 模拟：50 万样本 × 300 特征 ≈ 1.2GB
    n_samples, n_features = 500_000, 300
    print(f"\n构造 {n_samples:,} × {n_features} 特征矩阵 "
          f"({n_samples * n_features * 8 / 1024**3:.1f} GB)...")

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 3, n_samples)

    # 步骤 1：大矩阵不压缩保存（方便 mmap）
    X_path = os.path.join(tmp_dir, "X_large.joblib")
    joblib.dump(X, X_path, compress=0)
    del X  # 释放内存

    print(f"步骤 1: X 文件 {os.path.getsize(X_path) / 1024**3:.2f} GB (compress=0)")

    # 步骤 2：mmap 加载
    X_mmap = joblib.load(X_path, mmap_mode='r')
    print(f"步骤 2: mmap 加载, 类型={type(X_mmap).__name__}")

    # 步骤 3：增量训练
    model = SGDClassifier(loss='log_loss', max_iter=1, tol=None, random_state=42)
    batch_size = 50_000
    classes = np.unique(y)

    print(f"\n步骤 3: 增量训练 batch_size={batch_size:,}, "
          f"共 {n_samples // batch_size} 批...")
    t0 = time.perf_counter()

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        X_batch = np.array(X_mmap[start:end])      # 只复制当前 batch
        y_batch = y[start:end]
        model.partial_fit(X_batch, y_batch, classes=classes)

    t_train = time.perf_counter() - t0

    # 步骤 4：评估
    first_batch_acc = model.score(
        np.array(X_mmap[:batch_size]), y[:batch_size]
    )
    print(f"\n步骤 4: 评估")
    print(f"  训练耗时: {t_train:.1f}s")
    print(f"  准确率 (首 batch): {first_batch_acc:.4f}")

    # 步骤 5：保存模型
    model_path = os.path.join(model_dir, f"sgd_incremental_{TODAY}.joblib")
    joblib.dump(model, model_path, compress=3)
    print(f"  模型: {model_path} ({os.path.getsize(model_path)/1024:.1f} KB)")

    del X_mmap, y
    print(f"\n📖 核心：{n_samples * n_features * 4 / 1024**3:.1f}GB 特征矩阵在磁盘上")
    print(f"   训练过程只占用 batch_size × n_features ≈ {batch_size * n_features * 4 / 1024**2:.0f} MB 内存")
    print(f"\n✅ 场景 C 完成\n")


# ╔══════════════════════════════════════════════════════════════╗
# ║  汇总                                                        ║
# ╚══════════════════════════════════════════════════════════════╝

def summary():
    print("=" * 60)
    print("第 10 章 · 综合实战 — 全部场景完成")
    print("=" * 60)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    十条铁律回顾                              ║
╠══════════════════════════════════════════════════════════════╣
║  1. 内层 n_jobs=1 — 不要嵌套并行                             ║
║  2. 大数组用 mmap — 多进程共享，不复制                        ║
║  3. 压缩和 mmap 互斥 — 选一个                                ║
║  4. 文件名带版本号 — 永不覆盖                                ║
║  5. 缓存的函数是纯函数 — 无副作用                             ║
║  6. 任务内 try/except — 失败不丢结果                         ║
║  7. CPU 密集 loky，I/O 密集 threading                        ║
║  8. load() 只加载信任来源                                    ║
║  9. sklearn 版本对齐                                         ║
║ 10. 先串行调通，再改并行                                     ║
╚══════════════════════════════════════════════════════════════╝

🎉 恭喜！joblib 系统性学习完成。
   下一个动作：在你的项目里找一个最慢的 for 循环，
   加上 Parallel(n_jobs=-1) 和 @memory.cache。
""")


# ============================================================
if __name__ == "__main__":
    scenario_a_text_classification()
    scenario_b_image_features()
    scenario_c_incremental_training()
    summary()

    shutil.rmtree(tmp_dir)
