# 第 07 章 · Parallel + delayed — 简易并行

> **难度**：⭐⭐ | **核心**：`Parallel`, `delayed` | **一句话**：把普通 for 循环改成并行，一行不动业务逻辑

---

## 7.1 解决什么问题？

你有一个 for 循环，每次迭代独立、互不依赖，但串行跑要很久：

```python
# 100 个超参组合，每个训练 3 分钟 = 300 分钟
results = []
for params in param_grid:
    score = train_and_evaluate(params)   # 3 分钟
    results.append(score)
```

你的机器有 8 个 CPU 核心，但这段代码只用了一个——剩下 7 个在围观。

**Parallel + delayed 做的事**：把这个 for 循环的迭代分发到多个 CPU 核心上同时执行。

```python
from joblib import Parallel, delayed

# 只改了这一行——其他什么都不变
results = Parallel(n_jobs=8)(
    delayed(train_and_evaluate)(params) for params in param_grid
)
# 300 分钟 → ~38 分钟（8 核加速 8×）
```

---

## 7.2 基本用法

### 7.2.1 最简示例

```python
from joblib import Parallel, delayed
import time

def slow_square(x):
    time.sleep(1)        # 模拟耗时操作
    return x ** 2

# 串行：10 个任务 × 1 秒 = 10 秒
results = [slow_square(i) for i in range(10)]

# 并行：10 个任务 / 4 核 ≈ 3 秒
results = Parallel(n_jobs=4)(
    delayed(slow_square)(i) for i in range(10)
)
```

### 7.2.2 语法拆解

```python
results = Parallel(n_jobs=4)(                    # 创建并行执行器，4 个 worker
    delayed(slow_square)(i)                      # 把每次调用"延迟"成一个任务
    for i in range(10)                           # 生成 10 个任务
)
```

`delayed` 是关键——它不立即执行 `slow_square(i)`，而是把它包装成一个"待执行的任务"。`Parallel` 收集所有任务后，分发给 4 个 worker 进程同时执行。

### 7.2.3 n_jobs 参数

```python
Parallel(n_jobs=4)    # 用 4 个 CPU 核心
Parallel(n_jobs=-1)   # 用满所有核心（最常用）
Parallel(n_jobs=1)    # 就是串行（调试用）
Parallel(n_jobs=-2)   # 留 1 个核心给系统（all - 1）
```

**建议**：日常用 `n_jobs=-1`。调试时改成 `n_jobs=1`——单进程模式错误信息更清晰。

---

## 7.3 delayed 的作用

没有 `delayed` 的话，for 循环里的函数会立即执行：

```python
# ❌ 这样写不行——在 Parallel 启动前就串行执行完了
results = Parallel(n_jobs=4)(
    slow_square(i) for i in range(10)    # slow_square 立即执行！
)
```

`delayed` 把函数调用变成"任务描述"而不是"立即执行"：

```python
# delayed 包装后，这只是 10 个"待办任务"的描述
tasks = [delayed(slow_square)(i) for i in range(10)]
# tasks 里是 10 个任务对象，还没执行

# Parallel 统一调度执行
results = Parallel(n_jobs=4)(tasks)
```

**类比**：`delayed` 就像把邮件地址写在信封上——还没寄出。`Parallel` 是把这些信封一起投递到 4 个邮递员手上。

---

## 7.4 实战场景

### 场景 1：并行交叉验证

```python
from sklearn.model_selection import KFold
from joblib import Parallel, delayed

def train_one_fold(train_idx, test_idx, X, y, params):
    """训练一折，返回分数"""
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

kf = KFold(n_splits=10)
scores = Parallel(n_jobs=-1)(
    delayed(train_one_fold)(train_idx, test_idx, X, y, {'C': 1.0})
    for train_idx, test_idx in kf.split(X)
)
print(f"10 折平均: {np.mean(scores):.4f}")
```

### 场景 2：并行超参搜索

```python
param_grid = [
    {'C': 0.1, 'penalty': 'l1'},
    {'C': 1.0, 'penalty': 'l2'},
    # ... 50 个组合
]

results = Parallel(n_jobs=-1, verbose=10)(
    delayed(evaluate_params)(params, X_train, y_train, X_test, y_test)
    for params in param_grid
)

# 找出最佳参数
best_idx = np.argmax([r['score'] for r in results])
print(f"最佳参数: {param_grid[best_idx]}")
```

### 场景 3：批量特征提取

```python
def extract_features(file_path):
    """对单个文件提取特征"""
    data = pd.read_csv(file_path)
    return compute_statistics(data)

file_list = [f'data/part_{i}.csv' for i in range(200)]

features = Parallel(n_jobs=-1)(
    delayed(extract_features)(f) for f in file_list
)
# 200 个文件，8 核并行 → ~25 倍单核时间
```

### 场景 4：Memory + Parallel 组合（缓存已完成的）

```python
from joblib import Memory, Parallel, delayed

memory = Memory('./cache', verbose=0)

@memory.cache
def train_one(params, X, y):
    model = RandomForestClassifier(**params, n_jobs=1)  # ← 注意这个 n_jobs=1
    model.fit(X, y)
    return model.score(X, y)

# 第一次跑：全部并行执行
scores = Parallel(n_jobs=-1)(
    delayed(train_one)(p, X, y) for p in param_grid
)

# 调整了几个参数后再跑——之前算过的直接从缓存读，瞬间完成
scores = Parallel(n_jobs=-1)(
    delayed(train_one)(p, X, y) for p in param_grid
)
```

**特别注意**：被并行调用的函数里，内部模型的 `n_jobs` 要设为 `1`。否则会嵌套并行——外层开了 8 个进程，每个进程内的 RandomForest 又试图开 8 个进程，64 个进程互相抢 CPU，反而更慢。

---

## 7.5 verbose — 看进度

```python
# verbose=0: 静默
# verbose=10: 显示进度信息
# verbose=50: 显示每个任务的完成时间

Parallel(n_jobs=-1, verbose=10)(
    delayed(task)(i) for i in range(100)
)
# 输出类似：
# [Parallel(n_jobs=8)]: Using backend LokyBackend with 8 concurrent workers.
# [Parallel(n_jobs=8)]: Done   5 out of 100 | elapsed: 12.3s remaining: 234.0s
# [Parallel(n_jobs=8)]: Done 100 out of 100 | elapsed: 185.2s finished
```

---

## 7.6 结果收集

默认返回 list，结果顺序和输入顺序一致：

```python
results = Parallel(n_jobs=4)(
    delayed(slow_square)(i) for i in [3, 1, 4, 2]
)
# results = [9, 1, 16, 4] — 顺序保证
```

---

## 7.7 常见坑

### 坑 1：嵌套并行——把机器拖死

```python
# ❌ 外层 8 进程 × 内层 RandomForest 8 进程 = 64 进程混战
@memory.cache
def train(params, X, y):
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1)  # ← 内层也并行
    return model.fit(X, y).score(X, y)

Parallel(n_jobs=8)(delayed(train)(p, X, y) for p in grid)  # ← 外层也并行
```

```python
# ✅ 内层设 n_jobs=1，或者不用并行模型
def train(params, X, y):
    model = RandomForestClassifier(n_estimators=100, n_jobs=1)
    return model.fit(X, y).score(X, y)
```

### 坑 2：在延迟函数内修改全局可变对象

```python
# ❌ 子进程修改的是自己内存空间里的副本，主进程看不到
results_list = []
def bad_func(i):
    results_list.append(i)     # 子进程的 results_list，不是主进程的那个

Parallel(n_jobs=4)(delayed(bad_func)(i) for i in range(10))
print(results_list)  # [] — 空的！
```

```python
# ✅ 用 Parallel 的返回值
def good_func(i):
    return i

results = Parallel(n_jobs=4)(delayed(good_func)(i) for i in range(10))
print(results)  # [0, 1, 2, ..., 9] ✅
```

### 坑 3：传了超大数组给每个任务

```python
# ❌ 每个子进程都要序列化一个 5GB 的 X
Parallel(n_jobs=8)(
    delayed(train)(params, X, y)   # X 5GB → 每个子进程复制一份 → 40GB
    for params in grid
)
```

```python
# ✅ 用 mmap 共享 X
X_mmap = joblib.load('X.joblib', mmap_mode='r')
Parallel(n_jobs=8)(
    delayed(train)(params, X_mmap, y)  # 所有子进程共享同一份 mmap
    for params in grid
)
```

### 坑 4：lambda 函数传不进子进程

```python
# ❌ pickle 不能序列化 lambda（Windows 上尤其容易出问题）
Parallel(n_jobs=4)(
    delayed(lambda x: x**2)(i) for i in range(10)
)

# ✅ 用命名函数
def square(x): return x**2
Parallel(n_jobs=4)(delayed(square)(i) for i in range(10))
```

### 坑 5：传了大对象作为非迭代参数

```python
# ❌ big_data 每个任务都要传一遍
big_data = load_huge_file()
Parallel(n_jobs=8)(
    delayed(process)(big_data, idx) for idx in range(100)
)

# ✅ 用 mmap 或把大对象放在函数内部加载
def process(idx):
    big_data = load_huge_file()  # 每个子进程独立加载
    ...
```

---

## 7.8 什么时候用 Parallel，什么时候不用

| 用 Parallel | 不用 Parallel |
|-------------|---------------|
| 每次迭代独立，互不依赖 | 迭代之间有依赖（B 的输入是 A 的输出） |
| 每个任务耗时 > 0.1 秒 | 任务太短（<10ms），进程开销大于任务本身 |
| 任务数 > CPU 核心数 | 只有 2-3 个任务——不如直接 `ThreadPoolExecutor` |
| CPU 密集型（纯计算） | 需要共享大量 Python 对象（序列化开销大） |
| 需要加速、愿意接受一点开销 | 代码已经很快（<1 秒），不值得改造 |

---

## 7.9 本章要点

- [ ] `Parallel(n_jobs=-1)(delayed(fn)(args) for args in ...)` — 标准写法
- [ ] `delayed` 的作用：不立即执行，包装成"待办任务"
- [ ] `n_jobs=-1` 用满 CPU，`n_jobs=1` 调试用
- [ ] `verbose=10` 看进度
- [ ] 结果按输入顺序返回，和串行结果一致
- [ ] **最重要的一条**：被并行调用的函数内部不要再开并行（n_jobs=1）
- [ ] 大数组用 mmap 共享，不要每个子进程各传一份
- [ ] 不要用 lambda，不要改全局变量，不要嵌套并行
