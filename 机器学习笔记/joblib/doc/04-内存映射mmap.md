# 第 04 章 · 内存映射（mmap）

> **难度**：⭐⭐⭐ | **核心参数**：`mmap_mode` | **一句话**：大数据加载的终极方案——文件留在磁盘，用到哪读到哪

---

## 4.1 解决什么问题？

你做特征工程，生成了一个 30GB 的特征矩阵。你想加载它来训练模型：

```python
# ❌ 30GB 文件 → 30GB 内存 → 你的 16GB 机器直接 OOM
X = joblib.load('features.joblib')

# ✅ 文件留在磁盘，内存几乎不涨
X = joblib.load('features.joblib', mmap_mode='r')
```

**mmap 的核心思想**：不把整个文件搬进内存，而是建立一个"映射"——让你的程序看到一个"数组"，但这个数组的实际数据还在磁盘上。只有当你访问某个位置时，操作系统才去读那一小块。

---

## 4.2 三种模式

```python
# 只读映射（最常用）—— 能读不能改
X = joblib.load('features.joblib', mmap_mode='r')

# 读写映射 —— 修改会写回磁盘（小心！）
X = joblib.load('features.joblib', mmap_mode='r+')

# 写时复制 —— 修改只在内存生效，不影响磁盘
X = joblib.load('features.joblib', mmap_mode='c')
```

| 模式 | 能读 | 能改 | 修改影响磁盘？ | 使用场景 |
|------|------|------|--------------|---------|
| `'r'` | ✅ | ❌ | — | **加载特征、模型参数，只读使用** |
| `'r+'` | ✅ | ✅ | ✅ 会写回 | 需要对磁盘文件做原地修改 |
| `'c'` | ✅ | ✅ | ❌ 不影响 | 需要临时修改但不影响原文件 |

**90% 的情况用 `'r'` 就够了。**

---

## 4.3 实际效果

用代码说话——加载一个 160MB 的数组：

```python
import joblib
import numpy as np

# 准备数据
arr = np.random.randn(20_000_000).astype(np.float64)  # ~160 MB
joblib.dump(arr, 'big.joblib', compress=0)

# 普通加载 —— 全部读入内存，等了 0.5 秒
X1 = joblib.load('big.joblib')              # 耗时 0.5s，内存 +160 MB

# mmap 加载 —— 几乎瞬间，内存基本不涨
X2 = joblib.load('big.joblib', mmap_mode='r')  # 耗时 0.001s，内存 +0 MB
```

前者的 0.5 秒花在"把 160MB 数据从磁盘复制到内存"上。后者的 0.001 秒只是"在页表里记了一笔映射关系"。

当你真正访问数据时：

```python
# 首次访问 —— 触发 page fault，去磁盘读对应的 4KB 页
first_1000 = X2[:1000].copy()   # ~0.5ms

# 再次访问 —— 数据已在系统页缓存，不读盘
first_1000 = X2[:1000].copy()   # ~0.01ms
```

---

## 4.4 多进程共享——mmap 的隐藏大招

这是 mmap 最值钱但最容易被忽略的功能。

```python
# 主进程中 mmap 加载
X = joblib.load('features.joblib', mmap_mode='r')

# fork 出 8 个子进程做并行计算
from joblib import Parallel, delayed
results = Parallel(n_jobs=8)(
    delayed(process_chunk)(X, i) for i in range(8)
)
```

**关键**：8 个子进程看到的是**同一份物理内存页**，而不是各自复制 30GB。

不用 mmap 时，每个子进程都要独立序列化/反序列化一遍数据——8 个进程 = 30GB × 8 = 240GB 内存。用了 mmap，8 个进程共享同一份映射，总共只占用实际访问过的页面。

这是第 07-09 章讲 `Parallel` 时会反复用到的基础概念。

---

## 4.5 什么时候用、什么时候不用 — 实际应用中的边界

### 核心判断：mmap 主要用在特征矩阵/大数组，模型文件一般不需要

这是实际工作中最容易误判的地方——很多人学了 mmap 就什么文件都加 `mmap_mode='r'`。真实情况是：

```
需要 mmap 的                        不需要 mmap 的
──────────────────────────────────────────────────
特征矩阵 (100万 × 500)  → 4 GB       sklearn 模型   → 几 KB ~ 几十 MB
TF-IDF 矩阵             → 几十 GB     Pipeline       → 几 MB ~ 百来 MB
图像特征提取后的矩阵     → 几十 GB     神经网络权重    → 几十 MB ~ 几百 MB
文本嵌入向量矩阵         → 几 GB      元信息 dict     → 几 KB
```

### 为什么模型文件一般不需要 mmap？

绝大多数 sklearn 模型本质上存的是**模型参数**，不是训练数据，体积非常小：

| 模型 | 典型文件体积 | 原因 |
|------|------------|------|
| `LogisticRegression` | 几 KB ~ 几 MB | 只存系数矩阵 `coef_` 和截距 `intercept_` |
| `LinearRegression` / `Ridge` / `Lasso` | 几 KB ~ 几 MB | 同上，参数数量和特征数线性相关 |
| `SVC` (SVM) | 几 MB ~ 几十 MB | 存支持向量——样本数越多越大，但通常不会上 GB |
| `RandomForestClassifier` (100棵树) | 几 MB ~ 几十 MB | 存每棵树的节点分裂规则和叶子值 |
| `XGBoost` / `LightGBM` | 几 MB | 本身就是紧凑的二进制格式 |
| `MLPClassifier` (3层) | 几 MB ~ 几十 MB | 存各层的权重矩阵 |
| `KMeans` | 几 KB ~ 几 MB | 只存 k 个聚类中心 |
| **整个 `Pipeline`** (scaler + PCA + model) | 几 MB ~ 百来 MB | 预处理器参数 + 模型参数，仍远小于数据 |

这些模型文件大小在内存里完全不是问题。用 `joblib.load()` 直接全量加载，耗时不到 1 秒，内存涨几十 MB。加了 `mmap_mode='r'` 反而多余——模型本来就全量参与计算，映射的开销纯属浪费。

### 例外：哪些模型可能大到需要 mmap？

只有两类模型体积可能膨胀到 GB 级别：

| 模型 | 什么时候会变大 | 应对 |
|------|--------------|------|
| **k-NN 类** (`KNeighborsClassifier`) | 本质就是存了整个训练集——如果训练集 5GB，模型就 5GB | 用 `mmap_mode='r'`，或换 Approximate k-NN 库 |
| **超大树模型** | 几百棵很深的大树，或 ensemble 里存了几百个模型 | 通常仍不超过 1GB；如果超大则考虑 mmap |
| **深度学习模型** (PyTorch/TF) | 大语言模型可能几十 GB | 不用 joblib，用框架自己的 `state_dict` + `torch.save` |

但这些都是特殊情况。日常工作中：

> **经验法则：特征矩阵考虑 mmap，模型文件直接 load。**

### 要用 mmap 的场景

| 场景 | 为什么 |
|------|--------|
| **特征矩阵 > 500 MB** | 数据量是 mmap 的主战场 |
| 文件 > 可用内存的一半 | 避免 OOM |
| 只需要访问数组的一部分 | 只加载用到的页面 |
| 多进程共享同一份数据 | 不复制，共享物理内存页 |
| 文件在本地 SSD/NVMe | 随机访问性能好 |
| 加载速度比计算速度更重要 | mmap 加载几乎瞬时 |

### 不要用 mmap 的场景

| 场景 | 为什么 |
|------|--------|
| **模型文件**（sklearn Pipeline、单个 estimator） | 体积小（KB~百MB），直接 load 更快 |
| 文件 < 100 MB | 映射开销不值得 |
| 需要全部数据做密集计算 | 那无论如何都会全读到内存 |
| 文件在网络上（NFS、云存储） | mmap 不支持远程文件系统 |
| 需要频繁随机写入 | 每次写入触发磁盘 I/O，性能差 |

---

## 4.6 常见坑

### 坑 1：mmap 是只读的

```python
X = joblib.load('data.joblib', mmap_mode='r')
X[0, 0] = 1.0  # ❌ ValueError: read-only

# ✅ 如果需要修改，先复制到内存
X_copy = np.array(X)
X_copy[0, 0] = 1.0  # 可以
```

### 坑 2：压缩和 mmap 互斥

```python
# ❌ 压缩了的文件不能用 mmap——数据必须先解压才能映射
joblib.dump(X, 'f.joblib', compress=3)
X = joblib.load('f.joblib', mmap_mode='r')  # mmap_mode 被静默忽略

# ✅ 需要 mmap 就不压缩
joblib.dump(X, 'f.joblib', compress=0)
X = joblib.load('f.joblib', mmap_mode='r')  # mmap 生效
```

### 坑 3：关闭文件后 mmap 失效

```python
X = joblib.load('big.joblib', mmap_mode='r')
# ... 使用 X ...
# 只要 X 这个变量还在作用域内，映射就有效

# 如果删除了源文件，已经映射的数组不受影响（Linux 的 inode 机制）
# 但新进程无法再映射这个文件
```

### 坑 4：网络文件系统上的 mmap

```python
# ❌ 不要在网络路径上 mmap
X = joblib.load('//nfs_server/data.joblib', mmap_mode='r')
# 可能行为异常或极慢

# ✅ 先复制到本地再 mmap
import shutil
shutil.copy('//nfs_server/data.joblib', '/tmp/data.joblib')
X = joblib.load('/tmp/data.joblib', mmap_mode='r')
```

### 坑 5：忘了 mmap 加载的不是普通数组

```python
X = joblib.load('data.joblib', mmap_mode='r')
type(X)  # numpy.memmap，不是 numpy.ndarray

# 大部分操作一样，但要注意：
# - X 的修改受限（只读模式下不能改）
# - 不能改变 X 的 shape（reshape 会返回一个 view 而不是拷贝）
```

---

## 4.7 实战：大特征矩阵的训练流程

一个完整的例子——30GB 特征矩阵，用 mmap 加载 + 分批训练：

```python
from sklearn.linear_model import SGDClassifier
import joblib
import numpy as np

# 1. mmap 加载大特征矩阵（不占内存）
X = joblib.load('X_30GB.joblib', mmap_mode='r')
y = joblib.load('y.joblib')           # 标签很小，正常加载

# 2. 增量训练——一次只用一小批数据
model = SGDClassifier(loss='log_loss', random_state=42)
batch_size = 10000

for start in range(0, len(X), batch_size):
    end = min(start + batch_size, len(X))
    # X[start:end] 只触发对应页的 page fault
    X_batch = np.array(X[start:end])  # 复制到内存做计算
    y_batch = y[start:end]
    model.partial_fit(X_batch, y_batch, classes=np.unique(y))

# 3. 保存模型（模型本身不大，可以压缩）
joblib.dump(model, 'sgd_model.joblib', compress=3)
```

这个流程的妙处：30GB 的特征矩阵始终在磁盘上，训练过程只占用 `batch_size × n_features` 的内存。

### 4.7.1 什么是增量训练？

普通训练（`.fit()`）要求**一次性把全部数据加载到内存**，然后在整个数据集上计算梯度：

```
全部数据 → 加载到内存 → 一次 fit → 得到模型
```

增量训练（`.partial_fit()`）则是**一次只用一个 batch 的数据**，更新完模型参数后这个 batch 就可以丢弃：

```
batch1 → partial_fit → 更新模型 → 丢弃 batch1
batch2 → partial_fit → 更新模型 → 丢弃 batch2
batch3 → partial_fit → 更新模型 → 丢弃 batch3
...
```

两者的区别好比：

| | 普通训练 `.fit()` | 增量训练 `.partial_fit()` |
|------|------|------|
| 内存 | 需要装下全部数据 | 只需装下一个 batch |
| 训练方式 | 一遍过 | 逐批迭代 |
| 适用场景 | 数据能装进内存 | 数据太大装不下，或数据是流式到来的 |
| 收敛 | 一次到位 | 可能需要多轮（epoch） |

### 4.7.2 哪些 sklearn 模型支持增量训练？

只要模型类上有 `partial_fit` 方法，就支持增量训练。以下是常用模型：

| 类别 | 模型 | 备注 |
|------|------|------|
| **线性模型** | `SGDClassifier` | 分类，支持 hinge/log_loss 等 loss |
| | `SGDRegressor` | 回归 |
| | `PassiveAggressiveClassifier` | 在线学习分类 |
| | `PassiveAggressiveRegressor` | 在线学习回归 |
| | `Perceptron` | 最简单线性分类器 |
| **朴素贝叶斯** | `MultinomialNB` | 文本分类常用 |
| | `BernoulliNB` | 二值特征 |
| | `GaussianNB` | 连续特征（*） |
| | `ComplementNB` | 不平衡文本分类 |
| | `CategoricalNB` | 类别特征 |
| **神经网络** | `MLPClassifier` | 多层感知机分类 |
| | `MLPRegressor` | 多层感知机回归 |
| **聚类/降维** | `MiniBatchKMeans` | 小批量 K-Means |
| | `MiniBatchDictionaryLearning` | 字典学习 |
| | `MiniBatchNMF` | 非负矩阵分解 |
| | `IncrementalPCA` | 增量 PCA |
| **其他** | `KBinsDiscretizer` | 连续值分箱 |
| | `HashingVectorizer` | 文本哈希向量化 |

> \* `GaussianNB` 的 `partial_fit` 在一次调用中更新全局统计量，不需要 batch 循环——但仍然可以分批喂数据。

### 4.7.3 实际用法模式

```python
# 所有增量模型都遵循同样的模式：
model = SGDClassifier(loss='log_loss')

# 第一次调用必须传 classes 参数
model.partial_fit(X_batch1, y_batch1, classes=np.unique(y))

# 后续调用不需要 classes
for X_batch, y_batch in data_stream:
    model.partial_fit(X_batch, y_batch)

# 用起来和普通模型一样
preds = model.predict(X_test)
```

### 4.7.4 什么模型不支持增量训练？

**树模型（RandomForest、XGBoost、LightGBM、CatBoost）和 SVM 不支持 `partial_fit`。** 它们需要全量数据来构建树结构或求解二次规划问题。对于这些模型，处理大数据的方式是：

- 对数据**采样**（取一个能装进内存的子集训练）
- 用 `warm_start=True` 逐步增加树的数量（不是真正的增量学习，只是追加树）
- 切换到支持增量训练的模型（如 `SGDClassifier` 替代 `LogisticRegression`）

---

## 4.8 快速决策卡

```
我的场景：                                   → 怎么加载
─────────────────────────────────────────────────────
sklearn 模型 / Pipeline（几KB~百MB）            → joblib.load(path) 直接加载就行
特征矩阵 > 500 MB，只需要读                      → joblib.load(path, mmap_mode='r')
文件 > 可用内存，只需要读                        → joblib.load(path, mmap_mode='r')
需要修改数据但不影响原文件                        → joblib.load(path, mmap_mode='c')
多进程共享同一份数据                             → joblib.load(path, mmap_mode='r')
文件在 NFS/云存储上                             → 先 shutil.copy 到本地，再 mmap
文件已经压缩了                                  → 放弃 mmap，或者重存为 compress=0
```

---

## 4.9 本章要点

- [ ] `mmap_mode='r'` — 大文件标配，加载瞬间完成
- [ ] mmap 主要用在**特征矩阵/大数组**，模型文件（KB~百MB）直接 load 即可
- [ ] 数据实际在磁盘，访问时才按页加载（4KB/页）
- [ ] 多进程共享：fork 后所有子进程映射同一个文件，不复制
- [ ] 压缩和 mmap 互斥——需要 mmap 就不要压缩
- [ ] `'r'` 只读，`'r+'` 会写回磁盘，`'c'` 写时复制
- [ ] 网络文件系统上 mmap 不可靠——先复制到本地
- [ ] 增量训练 + mmap = 用 16GB 机器处理 100GB 数据集
