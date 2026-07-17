# 第 11 章：大规模预测

前面所有章节都在处理**一条**时间序列。但实际业务中——电商有百万个 SKU、连锁店有上千家门店、SaaS 有几百个客户。这一章讲如何把 Prophet 扩展到这个规模。

---

## 11.1 问题定义

你需要为 $N$ 条独立的时间序列分别生成预测。$N$ 可能是几十、几百、甚至几十万。

```
单条序列（前面 10 章）:           大规模（本章）:

  y                                 y₁  ╱╲  ╱╲
  │    ╱╲                           y₂  ╱  ╲╱
  │   ╱  ╲  ← 一个模型              y₃ ╱    ╲╱╲
  └─────────                         ...
                                   yₙ ╱╲╱╲╱╲
                                   N 条序列 × N 个模型
```

---

## 11.2 并行方案：现实中的主流选择

对 Prophet 来说，并行化的瓶颈不是 CPU，而是**序列化 (serialization)**。Prophet 拟合后的模型对象包含 Stan/C++ 编译产物，pickle 时又大又慢。因此选哪个并行框架，核心看它怎么处理对象序列化。

### 主流方案对比

| 方案 | 序列化方式 | 典型规模 | 优势 | 劣势 |
|------|-----------|---------|------|------|
| **joblib** | pickle | < 1,000 条 | scikit-learn 生态标准，API 极其简洁，一行 `Parallel` 搞定 | pickle 大对象慢；Windows 下需 `if __name__ == '__main__'` 保护 |
| **concurrent.futures** | pickle | < 1,000 条 | 标准库，零依赖；接口清晰 | 同上——pickle 是瓶颈 |
| **multiprocessing.Pool** | pickle | < 1,000 条 | 标准库，控制细粒度最高 | 同上；API 比前两个繁琐 |
| **pathos.multiprocessing** | **dill** | < 5,000 条 | dill 比 pickle 强——能序列化 lambda、闭包、更多 C++ 对象 | 第三方依赖；社区较小 |
| **Ray** | **Apache Arrow** | 1,000 ~ 100,000 条 | 零拷贝序列化、自动容错、有 Dashboard 监控 | 架构重，学习曲线陡 |
| **Dask** | pickle / cloudpickle | 1,000 ~ 50,000 条 | 与 pandas/numpy 深度集成；支持分布式 | 调度开销比 Ray 大；调试困难 |

### 实际使用情况

**joblib 用得最普遍。** 它和 scikit-learn 深度绑定，绝大多数数据科学团队的第一选择。只要你的序列数不超过几千条，joblib 就是正确答案——零学习成本，配合 `Parallel(n_jobs=-1, verbose=10)` 几行代码跑通。

**如果你的团队已经在用 Ray 做其他事情**（如 RL、模型 serving、超参搜索），那用 Ray 做 Prophet 并行是自然的延伸，不需要额外引入依赖。

**Dask** 适合已经在 pandas 上用 Dask DataFrame 的团队。

### Prophet 特有的性能瓶颈

无论用哪个框架，真正拖慢并行的是：

```
总时间 = CPU时间 + 序列化时间 + 进程间通信时间
          (固定)     (可能是瓶颈!)    (通常小)
```

**序列化是隐藏的瓶颈。** 如果你发现并行加速比远低于 CPU 核心数，多半是每个 worker 在反复序列化/反序列化大对象。优化方向：

```python
# 优化 1: 别在进程间传递模型对象
# ❌ 慢——pickle 整个 Prophet 模型
def forecast_with_model(model, df):
    return model.predict(model.make_future_dataframe(30))

# ✅ 快——只传参数字典 + DataFrame
def forecast_with_params(params, df):
    model = Prophet(**params)
    model.fit(df)
    return model.predict(model.make_future_dataframe(30))
```

```python
# 优化 2: 每个 worker 做一批序列，而不是每一条
# 减少进程创建/销毁的开销
def forecast_batch(store_ids, df_all):
    results = {}
    for sid in store_ids:
        df_s = df_all[df_all['store_id'] == sid]
        # ... fit and predict
    return results
```

---

## 11.3 代码：三种主流实现

### 11.3.1 joblib 写法（最推荐，< 1000 条序列）

```python
from joblib import Parallel, delayed
from prophet import Prophet

# 单个序列的拟合+预测函数
def fit_and_predict(sid, df_series, periods=90, params=None):
    if params is None:
        params = {'weekly_seasonality': True, 'yearly_seasonality': 10}

    model = Prophet(**params)
    model.fit(df_series[['ds', 'y']])
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return sid, forecast[['ds', 'yhat']].tail(periods)

# 准备任务列表
tasks = [(sid, group[['ds', 'y']].copy())
         for sid, group in df.groupby('store_id')]

# 一行并行
results = Parallel(n_jobs=-1, verbose=10)(
    delayed(fit_and_predict)(sid, df_s)
    for sid, df_s in tasks
)

# 合并结果
forecasts = {sid: fc for sid, fc in results}
```

### 11.3.2 ProcessPoolExecutor 写法（标准库方案）

```python
from concurrent.futures import ProcessPoolExecutor, as_completed

def fit_and_predict_mp(args):
    """被 pickle 序列化的函数——必须是模块级函数，不能是 lambda"""
    sid, df_series, periods, params = args
    model = Prophet(**(params or {}))
    model.fit(df_series[['ds', 'y']])
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return sid, forecast[['ds', 'yhat']].tail(periods)

tasks = [(sid, group[['ds', 'y']].copy(), 90, {})
         for sid, group in df.groupby('store_id')]

with ProcessPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(fit_and_predict_mp, t): t[0] for t in tasks}
    for f in as_completed(futures):
        sid, fc = f.result()
        forecasts[sid] = fc
```

### 11.3.3 Ray 写法（大规模场景）

```python
# uv pip install ray
import ray

@ray.remote
def fit_and_predict_ray(sid, df_series, periods=90):
    from prophet import Prophet  # 必须在 remote 函数内部 import
    model = Prophet(weekly_seasonality=True, yearly_seasonality=10)
    model.fit(df_series[['ds', 'y']])
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return sid, forecast[['ds', 'yhat']].tail(periods)

# 启动 Ray
ray.init(num_cpus=8)

# 提交所有任务
futures = [fit_and_predict_ray.remote(sid, group[['ds', 'y']])
           for sid, group in df.groupby('store_id')]

# 收集结果
forecasts = {sid: fc for sid, fc in ray.get(futures)}

ray.shutdown()
```

### 11.3.4 选择决策

```
你的序列数量？
  ├── < 100 条 → 串行就够了，别引入并行复杂度
  ├── 100 ~ 1,000 条 → joblib（已装 scikit-learn 就有）
  ├── 1,000 ~ 10,000 条 → joblib + 优化序列化 + 批量处理
  └── > 10,000 条 → Ray（或 Spark，如果集群已有）

团队已经在用什么？
  ├── 纯 Python 数据栈 → joblib
  ├── 已经在用 Ray → Ray
  └── 已经在用 Dask DataFrame → Dask
```

### 11.3.5 坑与规避

---

## 11.4 失败处理与监控

大规模建模必然有部分序列失败。需要做好容错：

```python
import traceback

def safe_fit_and_predict(store_id, df_series, periods):
    """带容错的单序列预测"""
    try:
        # 检查最基础的数据条件
        if len(df_series) < 14:
            return store_id, None, f"数据不足 ({len(df_series)} 天)"

        if df_series['y'].nunique() < 3:
            return store_id, None, "方差过小，无法建模"

        model = Prophet(weekly_seasonality=True)
        model.fit(df_series)
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        # 合理性检查
        final_forecast = forecast[['ds', 'yhat']].tail(periods)
        if final_forecast['yhat'].isna().any():
            return store_id, None, "预测结果含 NaN"

        if (final_forecast['yhat'] < 0).any() and df_series['y'].min() >= 0:
            return store_id, None, "预测含负值（但历史无非负）"

        return store_id, final_forecast, "OK"

    except Exception as e:
        return store_id, None, f"异常: {str(e)}"

# 统计结果
ok = sum(1 for _, _, status in results if status == 'OK')
failed = len(results) - ok
print(f"成功: {ok}, 失败: {failed} ({failed/len(results)*100:.1f}%)")
```

---

## 11.5 大规模预测的调参策略

当 N 很大时，不能为每条序列单独调参。策略如下：

### 方案 A：统一参数（最快）

所有序列用同一组参数——选几条典型序列调好，推广到全部。

### 方案 B：按类别分组调参（推荐）

```python
# 按序列特征分组
def classify_series(series):
    """根据序列特征返回参数组"""
    cv = series['y'].std() / series['y'].mean()  # 变异系数
    trend = (series['y'].iloc[-30:].mean() - series['y'].iloc[:30].mean()) / series['y'].mean()

    if cv < 0.2:
        return 'stable'      # 稳定序列 → 紧先验
    elif trend > 0.3:
        return 'growing'     # 高增长 → 灵活趋势
    else:
        return 'volatile'    # 波动大 → 灵活季节

param_sets = {
    'stable':   {'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 5.0},
    'growing':  {'changepoint_prior_scale': 0.5,  'seasonality_prior_scale': 10.0},
    'volatile': {'changepoint_prior_scale': 0.1,  'seasonality_prior_scale': 20.0},
}

category = classify_series(df_series)
model = Prophet(**param_sets[category])
```

### 方案 C：层次化先验（NeuralProphet / Hierarchical Prophet）

如果序列之间存在自然的层级关系（如：全国 → 省 → 市 → 门店），可以利用层级结构共享信息。Prophet 本身不支持，需要用专门的层次化预测工具或手动实现。

---

## 11.6 常见问题速查

| 问题 | 原因 | 解决 |
|------|------|------|
| 某条序列 `fit` 报错 | 数据量太少或方差为零 | 在建模前加前置检查 |
| 内存不足 | 同时加载太多序列的预测结果 | 分批处理、及时释放 DataFrame |
| 进程池卡死 | 子进程异常未正确传递 | 用 `timeout` 参数 + try/except |
| 预测结果不一致 | 相同数据的两次拟合有微小随机差异 | 设置 `np.random.seed()` |
| Windows 上多进程慢 | `spawn` 模式下每个进程重新 import Prophet | 减少 worker 数，或用 `threading` |

---

## 11.7 核心概念清单

| 概念 | 一句话理解 |
|------|-----------|
| **批量预测** | 对 N 条序列分别建模——不是多变量联合模型 |
| **串行 vs 并行** | N < 100 串行就行，更大才需要并行 |
| **关闭不确定性** | `uncertainty_samples=0` 是最大提速手段 |
| **容错设计** | 大规模下必然有失败——不 crash，不沉默忽略 |
| **分组调参** | 按序列特征分组，不能每条都单独调 |

---

下一章进入生产化部署——如何把模型从笔记本搬到线上。
