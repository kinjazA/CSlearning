# 第4章：scikit-learn 集成实战

## 本章目标
掌握 Optuna 与 sklearn 的三种集成方式，能够完成端到端的分类/回归任务超参优化。

---

## 4.1 三种集成方式

### 方式一：OptunaSearchCV（最简单）

Optuna 提供了类似 `GridSearchCV` 的 API：

```python
from optuna.integration import OptunaSearchCV

model = RandomForestClassifier()

param_distributions = {
    'n_estimators': optuna.distributions.IntDistribution(50, 500),
    'max_depth': optuna.distributions.IntDistribution(2, 32),
    'min_samples_split': optuna.distributions.FloatDistribution(0.01, 0.5),
}

optuna_search = OptunaSearchCV(
    model,
    param_distributions,
    n_trials=100,
    cv=5,
    scoring='accuracy',
    random_state=42
)

optuna_search.fit(X_train, y_train)
print(f"最佳参数: {optuna_search.best_params_}")
print(f"最佳分数: {optuna_search.best_score_}")
```

> **注意**：`OptunaSearchCV` 使用 `optuna.distributions` 模块定义搜索空间（分布对象），而**不是** `trial.suggest_*` 方法。这是因为 `OptunaSearchCV` 模仿 sklearn 的 `GridSearchCV` 风格——参数空间在外部定义，Optuna 内部生成 trial 并调用 `suggest_*`。
>
> `optuna.distributions` 提供以下分布类：
> - `FloatDistribution(low, high, log=False, step=None)` — 对应 `suggest_float`
> - `IntDistribution(low, high, log=False, step=1)` — 对应 `suggest_int`
> - `CategoricalDistribution(choices)` — 对应 `suggest_categorical`
>
> 在原生 `optimize()` 方式中**不需要**使用这些分布类——直接用 `trial.suggest_*` 即可。

**优点**：代码最少，与 sklearn 风格完全一致
**缺点**：不支持剪枝，搜索空间定义方式与原生 Optuna 不同

---

### 方式二：原生 optimize（推荐）

更灵活，支持剪枝，搜索空间定义更自然。这是**生产环境最常用的方式**。

#### 基本结构

```python
def objective(trial):
    # 1. 从 trial 获取超参数
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    max_depth = trial.suggest_int('max_depth', 2, 32)
    min_samples_split = trial.suggest_float('min_samples_split', 0.01, 0.5)

    # 2. 构建模型
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )

    # 3. 交叉验证评估
    score = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()

    # 4. 返回标量值（必须是单个 float）
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

#### 目标函数的编写规范

**① 数据怎么传入？—— 用闭包，不要用全局变量**

当数据来自外部（不在 `objective` 内加载），用闭包捕获是最安全和最灵活的方式：

```python
# ✅ 推荐：用工厂函数 + 闭包
def make_objective(X, y):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 2, 32),
        }
        model = RandomForestClassifier(**params, random_state=42)
        return cross_val_score(model, X, y, cv=5).mean()
    return objective

study.optimize(make_objective(X_train, y_train), n_trials=100)

# ❌ 避免：全局变量——多进程、分布式场景会出问题
X, y = load_data()
def objective(trial):  # X, y 来自外部作用域
    return cross_val_score(RandomForestClassifier(), X, y, cv=5).mean()
```

**② 随机种子要三处一致**

不同 trial 之间用**不同的种子**保证独立性，但整个采样链需要可复现：

```python
def objective(trial):
    seed = 42 + trial.number  # ← 每个 trial 不同，保证独立性

    model = RandomForestClassifier(
        n_estimators=trial.suggest_int('n_estimators', 50, 500),
        random_state=seed           # ① 模型种子
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)  # ② CV 种子

    score = cross_val_score(model, X, y, cv=cv).mean()
    return score

study = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42)     # ③ 采样器种子——保证整个搜索可复现
)
```

> **为什么 `trial.number` 而不是 `42`？** 如果所有 trial 用同一个种子，那同一个参数组合在每次 trial 中产生完全相同的 CV 分数，Optuna 无法判断哪个更好。用 `trial.number` 保证每个 trial 有不同的随机拆分，但整条种子链可追溯。

**③ 目标函数必须返回标量**

```python
# ✅ 正确
return cross_val_score(model, X, y, cv=5).mean()   # 单个 float

# ❌ 错误
return cross_val_score(model, X, y, cv=5)           # 返回了 5 个值！
scores = cross_val_score(model, X, y, cv=5)
return scores                                        # 也是数组
```

**④ 目标函数要自包含、无副作用**

每次调用 `objective(trial)` 都应该是**独立的**。不要在 trial 之间共享状态：

```python
# ❌ 错误：用全局变量跨 trial 共享状态
best_so_far = 0
def objective(trial):
    global best_so_far
    score = evaluate(trial)
    best_so_far = max(best_so_far, score)  # trial 之间互相干扰
    return score
```

#### 评分指标的选择

`cross_val_score` 的 `scoring` 参数决定了 Optuna 优化什么。选错 scoring 比选错模型更致命。

**分类任务**：

| scoring | 含义 | direction | 适用场景 |
|---------|------|-----------|---------|
| `'accuracy'` | 准确率 | `'maximize'` | 类别均衡时 |
| `'f1'` | F1（二分类） | `'maximize'` | 正负样本不均衡 |
| `'f1_macro'` | 宏平均 F1 | `'maximize'` | 多分类且不均衡 |
| `'roc_auc'` | AUC | `'maximize'` | 二分类，关注排序质量 |
| `'neg_log_loss'` | 负对数损失 | `'maximize'` | 需要概率校准 |

**回归任务**：

| scoring | 含义 | direction | 适用场景 |
|---------|------|-----------|---------|
| `'neg_mean_squared_error'` | 负 MSE | `'maximize'` | 通用回归 |
| `'neg_root_mean_squared_error'` | 负 RMSE | `'maximize'` | 需要与原单位同量纲 |
| `'neg_mean_absolute_error'` | 负 MAE | `'maximize'` | 对异常值鲁棒 |
| `'r2'` | R² | `'maximize'` | 需要可解释的拟合度 |

> **注意**：sklearn 约定所有 scorer 都是"越大越好"。所以 MSE 加了 `neg_` 前缀取负数。这意味着即使优化目标是"最小化 MSE"，`direction` 仍然填 `'maximize'`（因为 `neg_MSE` 越大 = MSE 越小）。
>
> 如果你觉得 `neg_` 前缀绕，可以改用 `direction='minimize'` + 手动计算 MSE：
> ```python
> from sklearn.metrics import mean_squared_error
> from sklearn.model_selection import cross_val_predict
> study = optuna.create_study(direction='minimize')
> y_pred = cross_val_predict(model, X, y, cv=5)
> return mean_squared_error(y, y_pred)  # 越小越好
> ```

#### 在同一 Study 中对比多个模型

这是原生 `optimize` 方式相比 `OptunaSearchCV` 的**最大优势**——你可以在一个搜索过程中同时决定"用什么模型"和"用什么参数"：

```python
def objective(trial):
    model_type = trial.suggest_categorical(
        'model', ['RandomForest', 'XGBoost', 'LogisticRegression']
    )

    if model_type == 'RandomForest':
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int('rf_n_estimators', 50, 500),
            max_depth=trial.suggest_int('rf_max_depth', 2, 32),
            random_state=42
        )
    elif model_type == 'XGBoost':
        model = xgb.XGBClassifier(
            n_estimators=trial.suggest_int('xgb_n_estimators', 50, 500),
            max_depth=trial.suggest_int('xgb_max_depth', 2, 10),
            learning_rate=trial.suggest_float('xgb_lr', 0.01, 0.3, log=True),
            random_state=42
        )
    else:
        model = LogisticRegression(
            C=trial.suggest_float('lr_C', 1e-3, 1e3, log=True),
            max_iter=2000,
            random_state=42
        )

    return cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
```

**查看各模型的表现**：

```python
df = study.trials_dataframe()
for model_name in ['RandomForest', 'XGBoost', 'LogisticRegression']:
    subset = df[df['params_model'] == model_name]
    if len(subset) > 0:
        print(f"{model_name}: 最佳={subset['value'].max():.4f}, "
              f"平均={subset['value'].mean():.4f} (n={len(subset)})")

print(f"\n胜出模型: {study.best_params.get('model')}")
```

> **注意**：Optuna 的 `study.best_params` 包含**所有**被采样过的参数（包括只在某些分支出现的）。如果某分支没用到 `rf_n_estimators`，`best_params` 里可能不存在这个键。读参数字典时用 `.get()` 防御性访问。

#### 实战注意事项

**① 大数据集：减少 CV 折数或子采样**

```python
def objective(trial):
    # 数据 > 10 万行 → 3-fold 足够，能省一半训练时间
    cv = 3 if len(X) > 100_000 else 5   # 下划线 _ 做分隔符，只是为了增强可读性

    # 或者每次 trial 随机子采样（适合快速探索阶段）
    n_sample = min(50_000, len(X))
    idx = np.random.choice(len(X), n_sample, replace=False)

    score = cross_val_score(model, X[idx], y[idx], cv=cv).mean()
    return score
```

**② 容错：别让一个失败 trial 中断全部搜索**

```python
study.optimize(
    objective,
    n_trials=100,
    catch=(ValueError, RuntimeError, np.linalg.LinAlgError)
)
# 出错的 trial 标记为 FAIL，Optuna 自动跳过，继续下一个
```

**③ 优化完成后：在独立测试集上验证**

> ⚠️ **CV 分数不是最终成绩！** CV 是用来比较 trial 的"内部排名"。最终模型必须在**完全没有参与搜索的测试集**上评估。

```python
study.optimize(objective, n_trials=100)

# 1. 查看最佳参数
print(f"CV 最佳分数: {study.best_value:.4f}")
print(f"最佳参数: {study.best_params}")

# 2. 用最佳参数 + 全量训练数据重新训练
best_model = RandomForestClassifier(
    n_estimators=study.best_params['n_estimators'],
    max_depth=study.best_params['max_depth'],
    random_state=42
)
best_model.fit(X_train, y_train)

# 3. 在测试集上评估（测试集从未被 Optuna 看到过！）
test_score = best_model.score(X_test, y_test)
print(f"测试集分数: {test_score:.4f}")

# 4. 保存模型和搜索记录
import joblib
joblib.dump(best_model, 'best_model.pkl')
study.trials_dataframe().to_csv('optuna_history.csv', index=False)
```

**④ 和 OptunaSearchCV 的选择指南**

| 场景 | 推荐 | 原因 |
|------|------|------|
| 快速优化单个 sklearn 模型 | `OptunaSearchCV` | 代码最少，sklearn 风格 |
| 需要剪枝 | 原生 `optimize` | `OptunaSearchCV` 不支持剪枝 |
| 对比多个不同模型 | 原生 `optimize` | `OptunaSearchCV` 一次只能一个模型 |
| 有预处理超参数（scaler 等） | 原生 `optimize` | 可自由组合 Pipeline |
| 部署到生产 | 原生 `optimize` | 更好的错误处理和控制力 |
| 团队协作、可读性优先 | `OptunaSearchCV` | sklearn 用户一眼就懂 |

**⑤ 常见反模式**

```python
# ❌ 反模式 1：在 objective 里修改外部变量
X_global = load_data()
def objective(trial):
    global X_global
    X_global = preprocess(X_global)  # 每次 trial 都在修改同一份数据！

# ❌ 反模式 2：忘记设置 random_state 导致结果不可复现
def objective(trial):
    model = RandomForestClassifier(...)  # 没有 random_state！
    return cross_val_score(model, X, y).mean()  # 每次结果不同

# ❌ 反模式 3：在 objective 里打印太多东西
def objective(trial):
    for i in range(100):
        print(f"Epoch {i}: loss = ...")  # 100 个 trial × 100 epoch = 10000 行输出
    return score
# ✅ 正确：用 callback 或只在关键节点打印
```

---

### 方式三：带剪枝的手动 CV（最高效）

结合第3章的剪枝技术，在 fold 级别做早停：

```python
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 200),
        'max_depth': trial.suggest_int('max_depth', 2, 32),
    }

    scores = []
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for step, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X[train_idx], y[train_idx])
        score = model.score(X[val_idx], y[val_idx])
        scores.append(score)

        # 每完成一个 fold 就报告并检查剪枝
        trial.report(np.mean(scores), step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(scores)
```

---

## 4.2 完整 Pipeline 示例

真实项目中，超参数搜索应该包含**数据预处理**和**模型训练**的完整流程：

```python
def objective(trial):
    # 1. 预处理超参数
    scaler_type = trial.suggest_categorical('scaler', ['standard', 'minmax', 'none'])
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = None

    # 2. 模型超参数
    model_type = trial.suggest_categorical('model', ['rf', 'gb', 'svm'])

    if model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int('rf_n_estimators', 50, 300),
            max_depth=trial.suggest_int('rf_max_depth', 3, 20),
        )
    elif model_type == 'gb':
        model = GradientBoostingClassifier(
            n_estimators=trial.suggest_int('gb_n_estimators', 50, 300),
            learning_rate=trial.suggest_float('gb_lr', 0.01, 0.3, log=True),
        )
    else:
        model = SVC(
            C=trial.suggest_float('svm_C', 1e-3, 1e3, log=True),
            kernel=trial.suggest_categorical('svm_kernel', ['rbf', 'linear']),
        )

    # 3. 构建 Pipeline
    steps = []
    if scaler:
        steps.append(('scaler', scaler))
    steps.append(('model', model))
    pipeline = Pipeline(steps)

    # 4. 交叉验证
    score = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy').mean()
    return score
```

---

## 4.3 回归任务

与分类任务几乎相同，只需调整评估指标：

```python
study = optuna.create_study(direction='minimize')  # 最小化 MSE

def objective(trial):
    # ... 定义模型 ...
    score = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error').mean()
    return score
```

---

## 4.4 本章实战任务

1. 用 `OptunaSearchCV` 快速优化一个随机森林（Iris 数据集）
2. 用原生 `optimize` 做同样的事情，对比代码差异
3. 添加剪枝，对比有/无剪枝的总耗时
4. 实现一个**多模型选择**的 Pipeline（RandomForest + GradientBoosting + SVM）

---

## 4.5 本章小结

- `OptunaSearchCV`：最简单，但不支持剪枝
- 原生 `optimize`：最灵活，推荐日常使用
- 手动 CV + 剪枝：最高效，适合大规模搜索
- Pipeline 可以包含预处理步骤，实现端到端优化

**下一步**：学习 PyTorch 中的 Optuna 集成。
