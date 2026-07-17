# 第 10 章 · ML/DL 综合实战

> **难度**：⭐⭐⭐⭐ | **前置**：第 01-09 章 | **一句话**：把 dump/load、Memory、Parallel 串成一个完整的机器学习工作流

---

## 10.1 三个实战场景

本章用一个完整的 ML 项目，把前面九章的知识点串起来。三个场景覆盖不同侧重点，你可以按需跳读：

| 场景 | 核心技巧 | 适合方向 |
|------|---------|---------|
| A · 文本分类 | Memory 缓存预处理 + Parallel 并行调参 + dump 持久化 | NLP、表格数据 |
| B · 图像特征 | mmap 大数组 + Parallel 批量提取 + 压缩存储 | CV、大规模特征 |
| C · 深度学习 | 缓存数据增强 + Parallel 超参搜索 + 模型保存 | DL 训练流程 |

---

## 10.2 场景 A · NLP 文本分类 Pipeline

### 10.2.1 完整代码

```python
import joblib
import numpy as np
from joblib import Memory, Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# ── 1. 设置缓存 ──
memory = Memory('./cache/nlp_pipeline', verbose=1)

# ── 2. 缓存：数据加载和预处理 ──
@memory.cache
def load_and_preprocess(file_path):
    """加载+清洗文本数据——同样的文件只跑一次"""
    import pandas as pd
    df = pd.read_csv(file_path)
    df['text'] = df['text'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
    return df['text'].values, df['label'].values

# ── 3. 缓存：向量化 ──
@memory.cache
def vectorize(texts, max_features=5000):
    """TF-IDF 向量化——同样的文本+参数只跑一次"""
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X = vec.fit_transform(texts)
    return X, vec

# ── 4. 并行：交叉验证评估 ──
def evaluate_model(X, y, alpha):
    """评估一组超参"""
    model = SGDClassifier(
        loss='log_loss', alpha=alpha, max_iter=1000,
        n_jobs=1, random_state=42          # ← 内层 n_jobs=1
    )
    scores = cross_val_score(model, X, y, cv=5)
    return {'alpha': alpha, 'mean': float(scores.mean()), 'std': float(scores.std())}

# ── 5. 并行：超参搜索 ──
def hyperparameter_search(X, y):
    alphas = [0.0001, 0.001, 0.01, 0.1]
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(evaluate_model)(X, y, a) for a in alphas
    )
    best = max(results, key=lambda r: r['mean'])
    return best

# ── 6. 持久化：保存最终模型 ──
def save_final_model(X, y, best_alpha, vec):
    """训练最终模型并保存"""
    model = SGDClassifier(
        loss='log_loss', alpha=best_alpha, max_iter=1000, random_state=42
    )
    model.fit(X, y)

    pipeline = Pipeline([('vec', vec), ('clf', model)])
    joblib.dump(pipeline, 'nlp_model_v1.joblib', compress=3)
    return pipeline

# ── 运行 ──
if __name__ == '__main__':
    texts, labels = load_and_preprocess('data/reviews.csv')  # 第一次：跑了
    # 下次重跑 → 缓存命中，瞬间
    X, vec = vectorize(texts, max_features=5000)             # 第一次：跑了

    best = hyperparameter_search(X, labels)
    print(f"最佳 alpha={best['alpha']}, CV={best['mean']:.4f}")

    pipeline = save_final_model(X, labels, best['alpha'], vec)

    # 部署时只需一行
    # pipeline = joblib.load('nlp_model_v1.joblib')
    # pipeline.predict(['this product is amazing'])
```

### 10.2.2 流程图

```
load_and_preprocess('reviews.csv')
       │  Memory 缓存 ── 第二次调用 0.001s
       ▼
  texts, labels
       │
       ▼
vectorize(texts)
       │  Memory 缓存 ── 同样参数不重算
       ▼
    X (稀疏矩阵)
       │
       ▼
Parallel(n_jobs=-1) ─── 4 个 alpha × 5 折 = 20 个任务并行
       │  evaluate_model(alpha=0.0001) ─┐
       │  evaluate_model(alpha=0.001)  ─┤ 同时跑
       │  evaluate_model(alpha=0.01)   ─┤
       │  evaluate_model(alpha=0.1)    ─┘
       ▼
  最佳 alpha
       │
       ▼
save_final_model → dump('nlp_model_v1.joblib', compress=3)
```

---

## 10.3 场景 B · 图像特征提取（大数组 + mmap）

```python
import joblib
import numpy as np
from joblib import Parallel, delayed

def extract_features(image_path):
    """对单张图片提取特征向量"""
    from PIL import Image
    img = Image.open(image_path).resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    # 模拟：用预训练模型提取特征
    return arr.mean(axis=(0, 1))  # 简化示意

# ── 1. 并行提取特征 ──
image_paths = [f'images/img_{i:05d}.jpg' for i in range(50000)]

features = Parallel(n_jobs=-1, verbose=10)(
    delayed(extract_features)(p) for p in image_paths
)

# ── 2. 保存特征矩阵（压缩） ──
X = np.array(features)  # 50000 × 3, 很小
joblib.dump(X, 'features.joblib', compress=('zstd', 3))
print(f"特征矩阵: {X.shape}, 文件: {os.path.getsize('features.joblib')/1024**2:.1f}MB")

# ── 3. 如果是超大规模特征（100万 × 2048 = 8GB） ──
# 不压缩，用 mmap 加载做增量训练
X_large = np.random.randn(1_000_000, 2048).astype(np.float32)
joblib.dump(X_large, 'X_large.joblib', compress=0)  # 不压缩，方便 mmap
# → 后续用 mmap_mode='r' 加载，配合 SGDClassifier.partial_fit 增量训练
```

---

## 10.4 场景 C · 深度学习训练流程

```python
import joblib
from joblib import Memory, Parallel, delayed

memory = Memory('./cache/dl_training', verbose=1)

# ── 1. 缓存：数据增强 ──
@memory.cache
def load_and_augment(data_dir, seed=42):
    """加载数据+增强——同样的 data_dir + seed 只跑一次"""
    # 实际中用 torchvision.transforms 或 tf.keras.preprocessing
    # 这里示意
    import numpy as np
    rng = np.random.RandomState(seed)
    X = np.load(f'{data_dir}/X.npy')
    # 增强：加噪声
    X_aug = X + rng.normal(0, 0.01, X.shape)
    return X_aug

# ── 2. 并行：超参数搜索 ──
def train_one_config(config, X, y):
    """训练一组超参数——注意内部 n_jobs=1"""
    # 实际中用 PyTorch / TensorFlow
    # model = build_model(config)
    # history = model.fit(X, y, ...)
    import time, numpy as np
    time.sleep(1)  # 模拟训练
    rng = np.random.RandomState(hash(str(config)) % 2**32)
    return {'config': config, 'val_acc': 0.8 + 0.1 * rng.random()}

configs = [
    {'lr': lr, 'batch_size': bs, 'layers': layers}
    for lr in [0.001, 0.01]
    for bs in [32, 64]
    for layers in [2, 3]
]  # 8 个组合

# ── 3. 并行搜索 ──
X = load_and_augment('data/train', seed=42)
y = np.load('data/train/y.npy')

results = Parallel(n_jobs=-1, verbose=10)(
    delayed(train_one_config)(cfg, X, y) for cfg in configs
)

best = max(results, key=lambda r: r['val_acc'])
print(f"最佳配置: {best['config']}, 验证准确率: {best['val_acc']:.4f}")

# ── 4. 保存最佳模型 + 搜索记录 ──
joblib.dump(best, 'best_config.joblib', compress=3)
joblib.dump(results, 'search_results.joblib', compress=3)
```

---

## 10.5 完整工作流的通用模板

不管什么项目，你都可以套这个模板：

```python
import joblib
from joblib import Memory, Parallel, delayed

# ── 0. 初始化 ──
memory = Memory('./cache', verbose=1)
TODAY = '20240625'

# ── 1. 缓存昂贵的预处理 ──
@memory.cache
def preprocess(raw_data_path):
    # ... 清洗、转换 ...
    return clean_data

# ── 2. 并行执行独立任务 ──
results = Parallel(n_jobs=-1, verbose=10)(
    delayed(one_task)(params, data)
    for params in param_list
)

# ── 3. 持久化结果 ──
joblib.dump(results, f'results_{TODAY}.joblib', compress=3)
joblib.dump(best_model, f'model_{TODAY}.joblib', compress=3)

# ── 4. 部署加载 ──
# model = joblib.load(f'model_{TODAY}.joblib')
# preds = model.predict(new_data)
```

---

## 10.6 十条铁律

综合前面九章的踩坑经验，这是你应该记住的：

| # | 铁律 |
|---|------|
| 1 | **内层 n_jobs=1** — 被 Parallel 调用的函数内部不要再开并行 |
| 2 | **大数组用 mmap** — 多进程共享，不复制，省内存 |
| 3 | **压缩和 mmap 互斥** — 选一个：要速度还是体积 |
| 4 | **文件名带版本号** — `model_v2.1_20240625.joblib`，永不覆盖 |
| 5 | **被缓存的函数是纯函数** — 不写文件、不发请求、不改全局变量 |
| 6 | **任务函数内 try/except** — 一个任务失败不影响其他 |
| 7 | **CPU 密集用 loky，I/O 密集用 threading** |
| 8 | **load() 只加载信任来源** — 它能执行任意代码 |
| 9 | **训练和部署的 sklearn 版本要对齐** |
| 10 | **先串行调通，再改成并行** — n_jobs=1 调试，n_jobs=-1 运行 |

---

## 10.7 学习完成

从第一章到第十章，你掌握了 joblib 的全部核心能力：

```
dump / load    → 模型和数据的存取
compress       → 压缩算法选型
mmap           → 大文件零拷贝加载
Memory         → 函数结果缓存
Parallel       → 并行加速（核心大招）
backend        → 多进程 vs 多线程
进阶技巧       → timeout / batch_size / 错误处理
```

**下一步**：在你自己的项目中挑一个最慢的 for 循环，加上 `Parallel(n_jobs=-1)` 和 `@memory.cache`，感受一下速度变化。第一个改造通常是最爽的。
