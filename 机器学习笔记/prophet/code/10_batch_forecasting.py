"""
第 11 章：大规模预测
====================
- 串行批量预测
- 多进程并行预测
- 容错与监控
- 按特征分组调参
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# ============================================================
# 0. 生成模拟数据 — 10 家门店 × 3 年日销售
# ============================================================
np.random.seed(42)
n_stores = 10
store_ids = [f'store_{chr(65+i)}' for i in range(n_stores)]

all_data = []
for sid in store_ids:
    dates = pd.date_range('2022-01-01', '2024-12-31', freq='D')
    n = len(dates)
    # 每家门店有不同的趋势和噪声水平
    base = 500 + hash(sid) % 1000
    trend = np.linspace(base, base * (1 + 0.1 * np.random.randn()), n)  # 随机年增长率
    weekly = 0.2 * base * np.sin(2 * np.pi * np.arange(n) / 7)
    yearly = 0.1 * base * np.sin(2 * np.pi * np.arange(n) / 365.25)
    noise_scale = 0.05 * base * (0.5 + np.random.random())
    noise = np.random.randn(n) * noise_scale

    df_s = pd.DataFrame({
        'ds': dates,
        'y': np.maximum(trend + weekly + yearly + noise, 10),
    })
    df_s['store_id'] = sid
    all_data.append(df_s)

df = pd.concat(all_data, ignore_index=True)
print(f"数据总量: {len(df)} 行, {df['store_id'].nunique()} 家门店")

# ============================================================
# 方案 1: 串行循环
# ============================================================
def forecast_single(df_series, periods=90):
    """对单条序列建模并预测"""
    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=10,
        uncertainty_samples=0,  # 大规模场景下关闭不确定性提速
    )
    model.fit(df_series)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(periods)

print("\n" + "=" * 50)
print("方案 1: 串行循环")
print("=" * 50)

t0 = time.time()
results_serial = {}
for sid, group in df.groupby('store_id'):
    results_serial[sid] = forecast_single(group[['ds', 'y']])
t_serial = time.time() - t0

print(f"  完成 {len(results_serial)} 家门店, 耗时 {t_serial:.1f}s")
print(f"  平均每店: {t_serial / len(results_serial):.1f}s")

# ============================================================
# 方案 2: 多进程并行
# ============================================================
def fit_and_predict_parallel(args):
    """进程安全的单序列预测函数"""
    sid, df_series, periods = args
    try:
        model = Prophet(
            weekly_seasonality=True,
            yearly_seasonality=10,
            uncertainty_samples=0,
        )
        model.fit(df_series)
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        return sid, forecast[['ds', 'yhat']].tail(periods), "OK"
    except Exception as e:
        return sid, None, str(e)

print("\n" + "=" * 50)
print("方案 2: 多进程并行")
print("=" * 50)

tasks = [(sid, group[['ds', 'y']].copy(), 90)
         for sid, group in df.groupby('store_id')]

n_workers = max(1, mp.cpu_count() - 2)
results_parallel = {}

t0 = time.time()
with ProcessPoolExecutor(max_workers=n_workers) as executor:
    futures_map = {executor.submit(fit_and_predict_parallel, t): t[0] for t in tasks}
    for future in as_completed(futures_map):
        sid, forecast, status = future.result()
        results_parallel[sid] = forecast
t_parallel = time.time() - t0

print(f"  完成 {len(results_parallel)} 家门店, 耗时 {t_parallel:.1f}s (workers={n_workers})")
print(f"  加速比: {t_serial / t_parallel:.1f}x")

# ============================================================
# 方案 3: 带容错的批量预测
# ============================================================
def safe_forecast(sid, df_series, periods=90):
    """带前置检查和容错的预测"""
    # 前置检查
    if len(df_series) < 30:
        return sid, None, f"数据不足 ({len(df_series)} 天)"

    if df_series['y'].std() < 1e-6:
        return sid, None, "方差过小"

    # 建模
    try:
        model = Prophet(
            weekly_seasonality=True,
            yearly_seasonality=10,
            uncertainty_samples=0,
        )
        model.fit(df_series)
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        result = forecast[['ds', 'yhat']].tail(periods)

        # 合理性检查
        if result['yhat'].isna().any():
            return sid, None, "预测含 NaN"

        return sid, result, "OK"

    except Exception as e:
        return sid, None, f"异常: {e}"

print("\n" + "=" * 50)
print("方案 3: 带容错的批量预测")
print("=" * 50)

results_safe = {}
statuses = {}

for sid, group in df.groupby('store_id'):
    _, forecast, status = safe_forecast(sid, group[['ds', 'y']])
    results_safe[sid] = forecast
    statuses[sid] = status

ok = sum(1 for s in statuses.values() if s == 'OK')
failed = len(statuses) - ok
print(f"  成功: {ok}, 失败: {failed}")

if failed > 0:
    print("  失败详情:")
    for sid, st in statuses.items():
        if st != 'OK':
            print(f"    {sid}: {st}")

# ============================================================
# 4. 合并结果展示
# ============================================================
print("\n" + "=" * 50)
print("预测结果汇总（未来90天平均预测值）")
print("=" * 50)

for sid in sorted(results_safe.keys()):
    if results_safe[sid] is not None:
        avg = results_safe[sid]['yhat'].mean()
        print(f"  {sid}: {avg:.0f}")
    else:
        print(f"  {sid}: 预测失败")

print("\n完成！")
