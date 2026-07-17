# 第 02 章 · dump & load — 高效序列化

> **难度**：⭐⭐ | **核心函数**：`dump()`, `load()` | **日常频率**：几乎每天

---

## 2.1 解决什么问题？

训练一个模型可能要几十分钟甚至几小时。训练完你不想下次再训一遍——你需要把模型"存"下来，下次直接"读"回来用。

Python 自带 `pickle` 能做这件事，但有两个痛点：

1. **慢** — 模型里大量 numpy 数组，pickle 逐个元素处理，大模型能等十几秒
2. **占内存** — 加载一个 2GB 的模型文件，内存要涨 2GB+，小机器直接 OOM

joblib 的 `dump` / `load` 就是来解决这两个问题的：**更快地存，更省内存地读**。

---

## 2.1.1 先认识 pickle — Python 标准库的序列化方案

在深入 joblib 之前，有必要先了解 `pickle`——它是 Python 自带的序列化模块，也是 joblib 的对照基准。理解了 pickle 的用法和局限，才能体会 joblib 到底优化了什么。

### pickle 是什么？

`pickle` 是 Python 标准库的一部分，**无需安装，开箱即用**。它的职责是把任意 Python 对象转换成字节流（序列化），存到磁盘或通过网络发送；然后再从字节流还原成 Python 对象（反序列化）。

```
Python 对象 ── pickle.dump() ──→ 字节流（文件/内存/网络）
字节流      ── pickle.load() ──→ Python 对象
```

### 核心 API

pickle 提供两对函数，分别对应「写文件」和「写内存」：

| 函数 | 方向 | 用途 |
|------|------|------|
| `pickle.dump(obj, file, protocol)` | 对象 → 文件 | 把对象写入磁盘文件 |
| `pickle.load(file)` | 文件 → 对象 | 从磁盘文件读取对象 |
| `pickle.dumps(obj, protocol)` | 对象 → bytes | 把对象序列化为内存中的字节串 |
| `pickle.loads(data)` | bytes → 对象 | 从字节串反序列化为对象 |

带 `s` 的版本操作的是内存中的 `bytes`，不带 `s` 的操作文件。

### 基本示例

```python
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ===== 方式一：文件操作 =====
model = RandomForestClassifier().fit(X_train, y_train)

# 保存到文件
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# 从文件加载
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# ===== 方式二：内存操作 =====
# 序列化为 bytes（可发给 Redis、HTTP、数据库 BLOB）
data = pickle.dumps(model)
print(f"序列化后大小: {len(data) / 1024:.1f} KB")

# 从 bytes 还原
model = pickle.loads(data)
```

### 核心参数：protocol（协议版本）

`pickle.dump(obj, file, protocol)` 的 `protocol` 是最关键的参数，决定了序列化的格式和兼容性：

| protocol | 引入版本 | 特点 |
|----------|---------|------|
| `0` | Python 1.0 | ASCII 文本格式，**人类可读**，但体积巨大。只用于调试 |
| `1` | Python 1.0 | 旧版二进制格式，兼容 Python 2 |
| `2` | Python 2.3 | 支持新式类，Python 2 时代的常用选择 |
| `3` | Python 3.0 | Python 3 默认，支持 bytes 对象。**不兼容 Python 2** |
| `4` | Python 3.4 | 支持大对象（>4GB）、pickle 优化。**Python 3.4+ 的推荐选择** |
| `5` | Python 3.8 | 支持 out-of-band 数据（让大 numpy 数组不经过 pickle 主流程，零拷贝传输）。**Python 3.8+ 可用** |

**实际工作中怎么选**：

```python
# 日常使用——让 Python 选最高协议
pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
# 等同于 protocol=5（Python 3.8+）或 protocol=4（Python 3.4-3.7）

# 需要兼容旧系统时
pickle.dump(model, f, protocol=4)  # Python 3.4+ 都能读

# 跨语言/调试——人类可读（体积巨大，不推荐日常用）
pickle.dump(model, f, protocol=0)
```

> **注意**：`pickle.load()` 不需要指定 protocol——它会自动检测。

### pickle 能序列化什么？

几乎任何 Python 对象都可以——函数、类、实例、numpy 数组、sklearn 模型、dict、list。但有硬性限制：

| ✅ 能序列化 | ❌ 不能序列化 |
|------------|-------------|
| 基本类型（int, str, float, bool, None） | 文件对象（`open()` 返回的） |
| 容器（list, dict, set, tuple） | 网络连接、数据库连接 |
| 函数、类、类的实例 | lambda 表达式（某些情况） |
| numpy 数组 | 生成器（generator） |
| sklearn 模型 / Pipeline | 线程、进程、锁 |
| 大多数自定义对象 | C 扩展对象（除非实现了 `__reduce__`） |

### pickle 的核心优势 — 什么时候用它而不是 joblib

| 场景 | 推荐 |
|------|------|
| 简单 Python 对象（dict、list、少量数据） | **pickle** — 零依赖，够快 |
| 需要最大兼容性（对方不一定装了 joblib） | **pickle** — Python 自带 |
| 通过网络发送对象（给 Redis、消息队列） | **pickle.dumps()** — 直接出 bytes |
| 需要人类可读（调试） | **pickle protocol=0** — ASCII 明文 |
| 存 sklearn 模型、大 numpy 数组 | **joblib** — 快 2-5×，文件小 2-4× |
| 需要压缩 | **joblib** — 内置 compress 参数 |
| 需要 mmap 零拷贝加载 | **joblib** — pickle 不支持 |

**一句话**：pickle 是 Python 世界的通用序列化语言；joblib 是科学计算/机器学习场景的专项加速器。两者不互斥——简单对象用 pickle，大数组/模型用 joblib。

### pickle 的安全警告

和 joblib 一样，`pickle.load()` 会执行任意代码。**永远不要 unpickle 不信任来源的数据**。如果需要安全序列化，用 `json`（只支持基本类型，但不执行代码）。

---

## 2.2 基本用法

```python
import joblib

# 保存
joblib.dump(model, 'model.joblib')

# 加载
model = joblib.load('model.joblib')
```

就这两行，覆盖你 90% 的使用场景。

### 2.2.1 可以存什么？

任何 Python 对象——sklearn 模型、numpy 数组、dict、list、Pipeline、自定义类实例。只要 pickle 能序列化的，joblib 都能存。

### 2.2.2 存到哪？

```python
# 磁盘文件（最常用）
joblib.dump(model, 'model.joblib')

# 内存（传给 Redis / HTTP 响应 / 数据库 BLOB）
from io import BytesIO
buf = BytesIO()
joblib.dump(model, buf)
buf.seek(0)
model = joblib.load(buf)
```

### 2.2.3 后缀名随便写

`.joblib`、`.pkl`、`.z`、`.gz` 都行，只是文件名不同，不影响内部格式。建议统一用 `.joblib`，一眼能认出来。

---

## 2.3 压缩 — 省磁盘空间

### 2.3.1 什么时候需要压缩？

模型文件超过几百 MB、或者要长期存档、或者要通过网络传输时，压缩能显著减小体积。

```python
joblib.dump(model, 'model.joblib', compress=3)  # 默认推荐
```

`compress=3` 能让你用少量 CPU 时间换 50-70% 的体积缩减，是性价比最高的选择。不压缩用 `compress=0`。

### 2.3.2 压缩算法怎么选

```python
# 默认 zlib，通用场景首选
joblib.dump(model, 'model.joblib', compress=3)

# 追求速度（频繁读写）
joblib.dump(model, 'model.joblib', compress=('lz4', 0))    # 需 uv add lz4

# 最佳平衡（推荐，如果装了 zstd）
joblib.dump(model, 'model.joblib', compress=('zstd', 3))    # 需 uv add zstandard

# 长期归档（不常读，追求最小体积）
joblib.dump(model, 'model.joblib', compress=('lzma', 9))
```

| 场景 | 推荐配置 | 原因 |
|------|---------|------|
| 日常开发 | `compress=3` | 速度/体积平衡，什么都不要装 |
| 生产环境，追求启动速度 | `compress=('lz4', 0)` | 写入几乎不花时间 |
| 归档存储 | `compress=('lzma', 9)` | 体积最小，不介意慢 |
| 最佳平衡 | `compress=('zstd', 3)` | 速度接近 lz4，压缩接近 lzma |

> 第 03 章会展开讲各算法的具体场景和 benchmark。

### 2.3.3 level 选多少？

level 越高 = CPU 投入越多 = 压缩越狠但越慢。**level 3 之后边际收益骤降**——从 3 升到 9，慢 3-5 倍但体积只多省 5-10%。所以默认是 3。

结论：**日常用 3，归档用 9，追速度用 0 或 lz4**。

---

## 2.4 mmap — 大文件的正确打开方式

### 2.4.1 问题

加载一个 80GB 的特征矩阵，`joblib.load()` 会把整个文件复制到内存。机器只有 32GB 内存 → OOM。

### 2.4.2 解决：mmap_mode

```python
# ❌ 全部加载到内存（80GB 文件 → 80GB 内存）
data = joblib.load('big_features.joblib')

# ✅ 内存映射——文件留在磁盘，用到哪读到哪
data = joblib.load('big_features.joblib', mmap_mode='r')
```

加一个 `mmap_mode='r'`，加载操作变成**瞬间完成**（只是建了个指针，不实际读数据）。只有你真正访问 `data[1000]` 时，操作系统才去磁盘取那一小块。

### 2.4.3 什么时候用

| 用 mmap | 不用 mmap |
|---------|-----------|
| 文件 > 可用内存的一半 | 文件 < 100 MB |
| 只需要访问数组的一部分 | 需要全部加载后密集计算 |
| 多进程共享同一份数据 | 文件不在本地磁盘 |
| **大特征矩阵**（几 GB ~ 几十 GB） | **模型文件**（几 KB ~ 百来 MB） |

> **实际工作中**：mmap 主要用在特征矩阵和大数组，sklearn 模型文件通常只有几 KB 到几十 MB，直接 `load()` 就行。详见第 04 章 4.5 节的完整分析。

### 2.4.4 三种模式

```python
mmap_mode='r'   # 只读（最常用）
mmap_mode='r+'  # 读写（修改会写回磁盘，小心！）
mmap_mode='c'   # 写时复制（修改只在内存生效，不影响文件）
```

**记住**：`'r'` 是只读的，试图修改会报错。如果要改，先 `np.array(data)` 复制到内存。

---

## 2.5 实战：sklearn 模型的标准处理流程

### 2.5.1 训练 → 保存 → 加载 → 预测

```python
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. 训练
model = RandomForestClassifier(n_estimators=200, n_jobs=-1)
model.fit(X_train, y_train)

# 2. 保存（带版本号，防覆盖）
joblib.dump(model, 'rf_v2.1_20240624.joblib', compress=3)

# 3. 加载
model = joblib.load('rf_v2.1_20240624.joblib')

# 4. 预测
preds = model.predict(X_test)
```

### 2.5.2 Pipeline 也一样

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier()),
])
pipe.fit(X_train, y_train)

joblib.dump(pipe, 'pipeline_v1.joblib', compress=3)
pipe = joblib.load('pipeline_v1.joblib')
```

Pipeline 保存的是整个流程——加载回来直接就能 predict，不需要重新 fit scaler。

### 2.5.3 同时保存模型和元信息（⭐ 生产环境的标配模式）

#### 解决什么问题？

你有没有遇到过这种情况：

- 模型目录里有 20 个 `model_v3_xxx.joblib` 文件，忘了哪个是最优的？
- 线上加载了一个模型，预测结果不对劲，后来发现特征列顺序不对——但模型文件本身看不出来它训练时用了哪些特征
- 想回滚到上一个版本，但不知道哪个文件对应哪个精度
- CI/CD 管道想自动判断新模型是否达标，达标才部署——但无法在不加载完整模型的情况下做判断

**核心矛盾**：模型文件可能几百 MB，你只想看一眼「这模型精度多少、用了哪些特征、什么版本」，却不得不把整个模型加载到内存里。

**解决方案**：把模型和元信息分开存——元信息文件只有几 KB，看一眼就能做判断。

#### 基础模式

```python
import joblib
from datetime import date

model = ...  # 训练好的模型

# 1. 保存模型本身
joblib.dump(model, f'model_v1_{date.today().isoformat()}.joblib', compress=3)

# 2. 保存元信息（几 KB，加载几乎零成本）
meta = {
    'version': '1.0',
    'date': date.today().isoformat(),
    'features': ['age', 'income', 'score'],
    'accuracy': 0.92,
}
joblib.dump(meta, f'model_v1_{date.today().isoformat()}_meta.joblib')
```

使用时：

```python
# 先加载元信息（几 KB，瞬间完成，内存几乎不涨）
meta = joblib.load('model_v1_2026-06-29_meta.joblib')
print(f"精度: {meta['accuracy']:.2%}, 特征: {meta['features']}")

# 满足条件才加载大模型
if meta['accuracy'] >= 0.90 and all(f in available_features for f in meta['features']):
    model = joblib.load('model_v1_2026-06-29.joblib', compress=3)
else:
    raise ValueError("模型不达标或特征不匹配，拒绝加载")
```

#### 元信息应该包含什么？

以下是生产环境的标准字段，按重要程度排列：

```python
meta = {
    # ── 必填：模型标识 ──
    'model_name': 'customer_churn_rf',
    'version': '2.1',
    'train_date': '2026-06-29',

    # ── 必填：性能指标 ──
    'metrics': {
        'accuracy': 0.923,
        'precision': 0.87,
        'recall': 0.85,
        'f1': 0.86,
        'roc_auc': 0.95,
    },

    # ── 必填：特征信息 ──
    'feature_names': ['age', 'income', 'score', 'tenure_days', 'num_products'],
    'feature_count': 5,
    'target_name': 'churn',
    'target_classes': [0, 1],

    # ── 必填：依赖版本（加载失败时救命的信息）──
    'python_version': '3.11.9',
    'sklearn_version': '1.5.0',
    'joblib_version': '1.5.3',
    'numpy_version': '1.26.4',
    'pandas_version': '2.2.0',

    # ── 推荐：模型参数 ──
    'model_type': 'RandomForestClassifier',
    'hyperparams': {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
    },

    # ── 推荐：数据来源 ──
    'training_data': 'data/processed/train_2026Q2.parquet',
    'training_samples': 150_000,
    'validation_split': 0.2,
    'random_seed': 42,

    # ── 选填：训练信息 ──
    'training_duration_seconds': 342.5,
    'trainer': 'zhangsan',
    'notes': '增加了 tenure_days 特征，精度提升 2 个点',
}
```

#### 为什么版本号很重要

`sklearn_version` 是最容易踩的坑——sklearn 1.5 保存的模型，在 sklearn 1.2 上加载很可能报错。把版本记在 meta 里，加载前就能判断：

```python
import sklearn

meta = joblib.load('model_v2_meta.joblib')

if meta['sklearn_version'] != sklearn.__version__:
    print(f"⚠️ 模型用 sklearn {meta['sklearn_version']} 保存，"
          f"当前环境是 {sklearn.__version__}，可能不兼容")
    # 自动创建对应版本的环境、或拒绝加载、或发告警
```

#### load-before-load 模式 — 在加载前做决策

这个模式在生产环境中有四种典型场景：

**场景一：模型目录浏览**

项目跑了上百次实验，模型目录杂乱。meta 文件让你一眼看到每个模型的关键信息，不用逐一加载：

```python
import glob

for meta_path in sorted(glob.glob('models/*_meta.joblib')):
    meta = joblib.load(meta_path)
    print(f"{meta['model_name']} v{meta['version']} "
          f"→ acc={meta['metrics']['accuracy']:.3f}, "
          f"date={meta['train_date']}")
```

**场景二：自动部署门禁**

CI/CD 流水线判断新训练的模型是否达到上线标准：

```python
def should_deploy(meta_path: str, min_accuracy: float = 0.90) -> tuple[bool, str]:
    meta = joblib.load(meta_path)
    if meta['metrics']['accuracy'] < min_accuracy:
        return False, f"精度 {meta['metrics']['accuracy']:.2%} < {min_accuracy:.2%}"
    # 检查特征是否齐全
    required = set(meta['feature_names'])
    if not required.issubset(current_features):
        return False, f"缺失特征: {required - set(current_features)}"
    return True, "通过"

ok, reason = should_deploy('latest_meta.joblib')
if ok:
    deploy(joblib.load('latest.joblib'))
```

**场景三：模型回滚**

线上出问题需要紧急回滚——先快速扫描历史 meta，找到最近一个达标的版本：

```python
def find_best_rollback(model_dir: str, min_acc: float) -> str | None:
    best = None
    for meta_path in sorted(glob.glob(f'{model_dir}/*_meta.joblib'), reverse=True):
        meta = joblib.load(meta_path)
        if meta['metrics']['accuracy'] >= min_acc:
            best = meta_path
            break  # 按时间倒序，第一个达标的就是最近的
    if best:
        model_name = best.replace('_meta.joblib', '.joblib')
        print(f"回滚到: {joblib.load(best)['version']}, "
              f"精度={joblib.load(best)['metrics']['accuracy']:.2%}")
        return model_name
    return None
```

**场景四：特征校验**

生产环境换了数据源，加载模型前先确认特征是否匹配——不加载模型就能判断：

```python
meta = joblib.load('model_meta.joblib')

# 检查当前数据源的列和训练时的特征是否一致
missing = set(meta['feature_names']) - set(df.columns)
extra = set(df.columns) - set(meta['feature_names'])

if missing:
    raise ValueError(f"当前数据缺失训练时用到的特征: {missing}")
if extra:
    print(f"⚠️ 当前数据多出了训练时没有的特征: {extra}（将被忽略）")
```

#### 注意事项

| 注意点 | 说明 |
|--------|------|
| **meta 不要压缩** | meta 就几 KB，压缩解压反而浪费时间 |
| **meta 和模型文件用相同的前缀** | `model_v1_2026-06-29.joblib` ↔ `model_v1_2026-06-29_meta.joblib`，一目了然 |
| **meta 在手，模型不动** | 通过 meta 发现模型不达标时，可以删掉对应的模型文件，避免磁盘浪费 |
| **不是一个文件，是两份文件** | 有风险：meta 和模型可能不同步（复制了模型忘了复制 meta，或反过来）。生产环境建议封装成工具函数，保存和加载都走同一入口 |

---

## 2.6 常见问题 & 避坑

### 坑 1：加载大文件导致 OOM

```python
# ❌ 80GB 文件直接 load → 内存炸
data = joblib.load('huge.joblib')

# ✅ 加 mmap_mode='r'
data = joblib.load('huge.joblib', mmap_mode='r')
```

### 坑 2：模型文件被覆盖

```python
# ❌ 每次训练覆盖同一个文件，历史模型丢失
joblib.dump(model, 'model.joblib')

# ✅ 文件名带版本号 + 日期
joblib.dump(model, f'model_v3_{date.today().isoformat()}.joblib')
```

### 坑 3：mmap 加载后试图修改

```python
data = joblib.load('features.joblib', mmap_mode='r')
data[0, 0] = 1.0  # ❌ ValueError: read-only

# ✅ 先复制到内存
data_copy = np.array(data)
data_copy[0, 0] = 1.0
```

### 坑 4：所有数据打一个包

```python
# ❌ 把 X_train (5GB) 和 y_train (50MB) 打包存——加载时必须全部读入
joblib.dump({'X': X_train, 'y': y_train}, 'data.joblib')

# ✅ 分别保存——需要哪个加载哪个，X 还可以用 mmap
joblib.dump(X_train, 'X_train.joblib', compress=3)
joblib.dump(y_train, 'y_train.joblib')
```

### 坑 5：加载不信任来源的文件

`joblib.load()` 可以执行任意代码——只加载你自己或团队保存的文件。千万不要加载用户上传的 `.joblib` 文件。

### 坑 6：不同 sklearn 版本之间的模型

sklearn 1.5 保存的模型，在 sklearn 1.2 上加载可能报错。**训练和部署用同一个 sklearn 版本**。

---

## 2.7 本章要点

- [ ] `dump(value, path)` 存，`load(path)` 读——覆蓋 90% 场景
- [ ] `compress=3` 是日常最优选择——体积减半，速度几乎不降
- [ ] `mmap_mode='r'` 是大文件救星——文件留在磁盘，按需读取
- [ ] 文件名加版本号 + 日期，防止覆盖
- [ ] 大数组单独存，方便选择性加载和 mmap
- [ ] 别加载不信任来源的文件
- [ ] 训练和部署的 sklearn 版本要对齐
