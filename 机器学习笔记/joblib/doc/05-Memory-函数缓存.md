# 第 05 章 · Memory — 函数结果缓存

> **难度**：⭐⭐ | **核心类**：`Memory` | **一句话**：给函数加一个装饰器，同样的输入直接返回缓存，不用重算

---

## 5.1 解决什么问题？

你做数据分析或建模时，工作流里总有一些**跑一次很贵、但输入输出很稳定**的步骤：

- 从数据库/文件加载原始数据，清洗、去重、规范化
- 特征工程：独热编码、TF-IDF、特征交叉
- 数据增强：图像旋转裁剪、文本分词
- 中间结果：交叉验证每一折的模型

每次重跑 notebook 或调试下游代码时，这些步骤都要重新执行——哪怕输入完全没变。一次预处理跑 15 分钟，一天改 5 次参数就要等一个多小时。

**Memory 做的事**：记住函数的"输入 → 输出"映射。下次同样的输入进来，直接返回上次算好的结果，函数体根本不执行。

```python
from joblib import Memory

memory = Memory('./cache_dir', verbose=1)

@memory.cache
def preprocess(raw_path):
    """耗时 20 分钟的清洗流程"""
    # ... 复杂的 ETL ...
    return clean_data

# 第一次调用：真的跑了 20 分钟
data = preprocess('2024_sales.csv')

# 第二次调用：0.001 秒返回缓存
data = preprocess('2024_sales.csv')  # 函数体根本没执行
```

---

## 5.2 基本用法

### 5.2.1 三步上手

```python
from joblib import Memory

# 1. 指定缓存存放目录
memory = Memory('./cache', verbose=1)

# 2. 用装饰器标记要缓存的函数
@memory.cache
def expensive_task(param1, param2):
    # 耗时操作...
    return result

# 3. 正常调用——缓存自动生效
result = expensive_task(1, 'hello')  # 第一次：真的执行
result = expensive_task(1, 'hello')  # 第二次：读缓存
```

### 5.2.2 verbose 参数——看缓存命中了没

```python
memory = Memory('./cache', verbose=1)

@memory.cache
def compute(x):
    return x ** 2

compute(5)  # 输出: [Memory] Calling compute...
            #       compute(...) - 0.0s
compute(5)  # 输出: [Memory] 0.000s = cached
            #       (没有 "Calling"，说明没执行函数体)
```

`verbose=0` 是静默模式，`verbose=1` 告诉你每次是"真跑了"还是"读缓存了"，`verbose=10` 会打印更多细节（适合排查为什么不命中）。

---

## 5.3 缓存怎么判断"同样的输入"？

Memory 自动为**函数名 + 参数值**生成一个哈希值作为缓存的 key。底层逻辑：

```
cache_key = hash(
    函数源代码  +  参数值
)
```

### 5.3.1 什么情况命中缓存

```python
@memory.cache
def add(a, b):
    return a + b

add(1, 2)     # 执行 → 缓存
add(1, 2)     # 命中 ✓（参数一样）
add(a=1, b=2) # 命中 ✓（关键字参数也一样）
add(1, 3)     # 不命中 ✗（参数不一样）
```

### 5.3.2 什么情况不命中——改了函数体

```python
@memory.cache
def add(a, b):
    return a + b

add(1, 2)  # 执行 → 缓存

# 你改了函数实现（哪怕只改注释，源代码变了哈希就变）
@memory.cache
def add(a, b):
    """求和"""          # ← 加了注释
    return a + b

add(1, 2)  # 不命中！函数源代码变了，旧缓存失效
```

**设计意图**：如果你改了函数逻辑，旧缓存不应该被复用——否则你会拿到过时的结果。这是一个安全机制。

### 5.3.3 不可哈希的参数会怎样

numpy 数组、list 等可变对象不能直接哈希。Memory 会自动处理——它对数组内容计算哈希值而不是用对象 id：

```python
import numpy as np

@memory.cache
def process(arr):
    return arr.mean()

a1 = np.array([1, 2, 3])
a2 = np.array([1, 2, 3])

process(a1)  # 执行
process(a2)  # 命中 ✓（数组内容一样，即使是不同对象）
```

---

## 5.4 实战场景

### 场景 1：缓存数据预处理

```python
from joblib import Memory

memory = Memory('./preprocess_cache', verbose=1)

@memory.cache
def load_and_clean(file_path):
    """从 CSV 加载 + 缺失值填充 + 异常值处理 + 类型转换"""
    import pandas as pd
    df = pd.read_csv(file_path)
    # ... 30 行清洗代码 ...
    return df

# 同样的文件路径，不管你调用多少次，只执行一次
train = load_and_clean('data/train_2024.csv')
test = load_and_clean('data/test_2024.csv')
train_again = load_and_clean('data/train_2024.csv')  # ← 读缓存
```

### 场景 2：缓存特征工程

```python
@memory.cache
def engineer_features(df_cleaned):
    """特征构造：交叉、聚合、编码"""
    X = df_cleaned.copy()
    X['age_group'] = pd.cut(X['age'], bins=[0, 18, 35, 60, 100])
    X['income_per_age'] = X['income'] / X['age']
    # ... 20 行特征工程 ...
    return X

X = engineer_features(train)     # 第一次：跑了
X = engineer_features(train)     # 第二次：缓存命中，瞬间
```

**关键好处**：你在调模型参数时，每次重跑 notebook，前置的预处理和特征工程全部从缓存读——原本 30 分钟的等待变成 0.1 秒。

### 场景 3：调试时跳过中间步骤

```python
@memory.cache
def step1(raw_data):
    return heavy_operation_1(raw_data)   # 15 分钟

@memory.cache
def step2(step1_output):
    return heavy_operation_2(step1_output)  # 10 分钟

@memory.cache
def step3(step2_output):
    return heavy_operation_3(step2_output)  # 5 分钟

# 你在改 step3 的逻辑
# step1 和 step2 的输入没变 → 直接从缓存读
# 只有 step3 重新执行
result = step3(step2(step1(raw_data)))
#              ↑ 缓存    ↑ 缓存    ↑ 只有这个真跑了
```

这就是 Memory 最实用的场景——**让你在 pipeline 的任意位置"断点续跑"，只重算改了的部分**。

---

## 5.5 管理缓存

### 5.5.1 查看缓存目录

```python
memory = Memory('./cache')

# 缓存目录结构：
# ./cache/
#   joblib/
#     my_module/
#       my_function/
#         func_code_hash/          ← 函数源代码的哈希
#           input_args_hash/       ← 参数值的哈希
#             output.pkl           ← 缓存的返回值
```

### 5.5.2 清空缓存

```python
memory.clear()  # 删除所有缓存
# 或直接手动删除缓存目录
```

什么时候清：函数逻辑大幅改动后，旧缓存没用了；或者磁盘空间紧张。

### 5.5.3 定期清理过期缓存

```python
from datetime import timedelta
memory.reduce_size(age_limit=timedelta(days=30))  # 删除 30 天前的缓存
```

---

## 5.6 Memory 和 Parallel 的互动（预告）

Memory 单独用已经很有价值，但和 `Parallel` 结合才是真正的大杀器。第 07 章会展开，这里先看一眼：

```python
from joblib import Memory, Parallel, delayed

memory = Memory('./cache', verbose=1)

@memory.cache
def train_one_fold(params, fold_id, X, y):
    """训练一折——结果缓存下来"""
    # ... 训练 ...
    return score

# 100 个参数组合 × 5 折 = 500 次训练
# 第一次跑：500 次全执行（并行加速）
# 第二次调参后重跑：之前算过的 fold 全部从缓存读
scores = Parallel(n_jobs=-1)(
    delayed(train_one_fold)(p, fid, X, y)
    for p in param_grid
    for fid in range(5)
)
```

---

## 5.7 常见坑

### 坑 1：函数有副作用

```python
@memory.cache
def train_and_save_model(data_path):
    model = train(data_path)
    # 副作用：把模型写到了磁盘
    model.to_json('model.json')      # ← 读缓存时这行不会执行！
    return model

# 第一次：训练 + 保存文件 ✓
# 第二次：从缓存读，model.json 不会被更新 ✗
```

**规则**：被缓存的函数应该是**纯函数**——同样的输入总是产生同样的输出，不依赖外部状态，不产生副作用（写文件、发请求、更新数据库）。

### 坑 2：缓存了可变对象

```python
@memory.cache
def get_config():
    return {'lr': 0.01, 'epochs': 10}

cfg = get_config()
cfg['lr'] = 0.1           # 修改了缓存的返回值对象！
cfg2 = get_config()       # 读缓存，但 cfg2 可能也被污染了
```

**规则**：缓存函数返回的对象不要原地修改。如果需要改，先 copy。

### 坑 3：忘记清缓存导致结果"没变"

```python
@memory.cache
def preprocess(data):
    return data.dropna()

# 你改了数据源的列名，但函数名和参数没变
# → Memory 认为输入没变 → 返回旧缓存
# → 你看到的还是旧列名的数据
```

**规则**：改了数据源或上游逻辑后，要么改函数名，要么清缓存。

### 坑 4：把 Memory 放在会被多次 import 的模块顶层

```python
# my_utils.py
memory = Memory('./cache')  # ← 每次 import 都会创建新的 Memory 实例

@memory.cache
def my_func(x):
    return x
```

这通常没问题（Memory 是幂等的），但在某些打包/部署场景可能路径错乱。建议把缓存目录路径写成配置项，而非硬编码。

---

## 5.8 本章要点

- [ ] `@memory.cache` — 装饰器一加，同样输入自动走缓存
- [ ] 缓存依据：函数源代码 + 参数值 → 改了函数体或参数才会重新执行
- [ ] 典型场景：缓存数据预处理、特征工程、中间结果——调试时只重算改过的部分
- [ ] `memory.clear()` 清缓存，`memory.reduce_size()` 清理过期缓存
- [ ] 被缓存的函数应该是纯函数——没有副作用，不依赖外部可变状态
- [ ] Memory + Parallel 组合 = 并行遍历参数时自动跳过已完成的组合
