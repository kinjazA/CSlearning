"""
第 10 章：数据预处理与异常处理
==============================
- 数据质量检查清单
- 缺失值诊断
- 异常值检测（IQR + 滚动窗口 + Prophet 残差）
- 异常值处理策略对比
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

# ============================================================
# 0. 生成"脏"数据
# ============================================================
np.random.seed(42)
dates = pd.date_range('2021-01-01', '2024-12-31', freq='D')
n = len(dates)

# 真实信号
trend = np.linspace(100, 300, n)
weekly = 50 * np.sin(2 * np.pi * np.arange(n) / 7)
yearly = 40 * np.sin(2 * np.pi * np.arange(n) / 365.25)
noise = np.random.randn(n) * 15
y_clean = trend + weekly + yearly + noise

# 人为制造问题
y_dirty = y_clean.copy()

# 1. 缺失值（随机 3%）
missing_idx = np.random.choice(n, size=int(n * 0.03), replace=False)
y_dirty[missing_idx] = np.nan

# 2. 缺失段（连续 2 周）
y_dirty[500:514] = np.nan

# 3. 异常值（数据错误 — 突然掉到 0）
error_idx = np.random.choice(n, size=15, replace=False)
y_dirty[error_idx] = 0

# 4. 真实极端事件（促销日 — 不应删除）
promo_idx = np.array([200, 400, 600, 800, 1000])
y_dirty[promo_idx] = 400

df = pd.DataFrame({'ds': dates, 'y_raw': y_dirty})
print(f"原始数据: {len(df)} 行")

# ============================================================
# 1. 数据质量检查清单
# ============================================================
def data_quality_report(df, y_col='y_raw'):
    """建模前必跑的数据质量检查"""
    df = df.copy()
    df['ds'] = pd.to_datetime(df['ds'])

    print("=" * 50)
    print("数据质量报告")
    print("=" * 50)

    # 1.1 基本信息
    total_days = (df['ds'].max() - df['ds'].min()).days
    print(f"\n1. 数据量: {len(df)} 行")
    print(f"   日期范围: {df['ds'].min().date()} ~ {df['ds'].max().date()} ({total_days} 天)")

    # 1.2 缺失检查
    full_days = pd.date_range(df['ds'].min(), df['ds'].max(), freq='D')
    missing_n = df[y_col].isna().sum()
    missing_dates = df[df[y_col].isna()]['ds']
    print(f"\n2. 缺失值: {missing_n} ({missing_n/len(df)*100:.1f}%)")

    # 找连续缺失段
    if len(missing_dates) > 0:
        gaps = (missing_dates.diff() > pd.Timedelta(days=1)).cumsum()
        for gap_id, group in missing_dates.groupby(gaps):
            if len(group) >= 3:
                print(f"   ⚠️  连续缺失段: {group.iloc[0].date()} ~ {group.iloc[-1].date()} ({len(group)} 天)")

    # 1.3 数值统计
    valid_y = df[y_col].dropna()
    print(f"\n3. y 统计:")
    print(f"   Min: {valid_y.min():.1f}, Max: {valid_y.max():.1f}")
    print(f"   Mean: {valid_y.mean():.1f}, Std: {valid_y.std():.1f}")
    print(f"   零值: {(valid_y == 0).sum()}, 负值: {(valid_y < 0).sum()}")

    # 1.4 异常值 (IQR)
    q1, q3 = valid_y.quantile(0.25), valid_y.quantile(0.75)
    iqr = q3 - q1
    n_outliers_iqr = ((valid_y < q1 - 3*iqr) | (valid_y > q3 + 3*iqr)).sum()
    print(f"\n4. 异常值 (3×IQR): {n_outliers_iqr} ({n_outliers_iqr/len(valid_y)*100:.1f}%)")

    # 1.5 趋势检查
    half = len(valid_y) // 2
    first_half = valid_y.iloc[:half].mean()
    second_half = valid_y.iloc[half:].mean()
    change_pct = (second_half - first_half) / abs(first_half) * 100
    print(f"\n5. 趋势变化: 前半段均值={first_half:.1f}, 后半段={second_half:.1f}, 变化={change_pct:.1f}%")

    # 1.6 季节性检查
    df['_wday'] = df['ds'].dt.dayofweek
    df['_month'] = df['ds'].dt.month
    valid_df = df.dropna(subset=[y_col])
    wday_cv = valid_df.groupby('_wday')[y_col].mean().std() / valid_y.mean()
    month_cv = valid_df.groupby('_month')[y_col].mean().std() / valid_y.mean()
    print(f"\n6. 季节性检查:")
    print(f"   周内变异系数: {wday_cv:.3f} ({'>0.1=存在' if wday_cv > 0.1 else '<0.1=弱'})")
    print(f"   月间变异系数: {month_cv:.3f} ({'>0.1=存在' if month_cv > 0.1 else '<0.1=弱'})")

    # 1.7 建模建议
    print(f"\n7. 建议:")
    if total_days < 365:
        print("   ⚠️ 数据不足 1 年，年季节性不可靠")
    if missing_n > len(df) * 0.1:
        print("   ⚠️ 缺失 > 10%，检查数据管道")
    if valid_y.max() / (valid_y.min() + 1) > 10:
        print("   💡 数值跨度 > 10 倍，考虑乘性季节性")
    if n_outliers_iqr > len(valid_y) * 0.03:
        print("   ⚠️ 异常值 > 3%，建模前需处理")

    print("=" * 50)
    return

data_quality_report(df)

# ============================================================
# 2. 异常值检测
# ============================================================
y = df['y_raw'].copy()

# 方法 A: IQR
q1, q3 = y.quantile(0.25), y.quantile(0.75)
iqr = q3 - q1
outliers_iqr = (y < q1 - 3*iqr) | (y > q3 + 3*iqr)

# 方法 B: 滚动窗口
rolling_mean = y.rolling(30, center=True).mean()
rolling_std = y.rolling(30, center=True).std()
outliers_rolling = np.abs(y - rolling_mean) > 4 * rolling_std

# 方法 C: Prophet 残差
temp_df = df.dropna(subset=['y_raw']).rename(columns={'y_raw': 'y'})
m = Prophet()
m.fit(temp_df)
fc = m.predict(temp_df[['ds']])
residuals = temp_df['y'] - fc['yhat']
outliers_prophet = np.abs(residuals) > 3 * residuals.std()

# 综合投票
valid_idx = ~y.isna()
outliers_combined = pd.Series(False, index=df.index)
outliers_combined[valid_idx] = (
    outliers_iqr[valid_idx].astype(int) +
    outliers_rolling[valid_idx].astype(int) +
    pd.Series(outliers_prophet.values, index=temp_df.index).reindex(valid_idx[valid_idx].index).fillna(0).astype(int)
) >= 2

print(f"\n异常值检测结果:")
print(f"  IQR 方法:      {outliers_iqr.sum()} 个")
print(f"  滚动窗口方法:  {outliers_rolling.sum()} 个")
print(f"  Prophet 方法:  {outliers_prophet.sum()} 个")
print(f"  综合投票(≥2):  {outliers_combined.sum()} 个")

# ============================================================
# 3. 异常值处理策略对比
# ============================================================
# 策略 A: 不做处理
df_a = df.rename(columns={'y_raw': 'y'}).dropna(subset=['y'])

# 策略 B: 标记为 NA（推荐）
df_b = df.rename(columns={'y_raw': 'y'}).copy()
df_b.loc[outliers_combined, 'y'] = None

# 策略 C: 用滚动中位数替换
df_c = df.rename(columns={'y_raw': 'y'}).copy()
rolling_median = df_c['y'].rolling(30, center=True, min_periods=5).median()
df_c.loc[outliers_combined, 'y'] = rolling_median[outliers_combined]

# 对比三种策略的预测效果
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
for ax, (label, df_proc) in zip(axes, [
    ('不处理异常值', df_a),
    ('标记为 NA (Prophet 自动平滑)', df_b),
    ('滚动中位数替换', df_c),
]):
    m = Prophet(weekly_seasonality=False, yearly_seasonality=False)
    m.fit(df_proc)
    fc = m.predict(m.make_future_dataframe(90))
    ax.plot(df_proc['ds'], df_proc['y'], '.', alpha=0.3, markersize=2, color='gray')
    ax.plot(fc['ds'], fc['yhat'], 'b-', lw=1.5)
    ax.set_title(label)

plt.suptitle('异常值处理策略对比', fontsize=14)
plt.tight_layout()
plt.show()

print("\n处理完成 — 查看上方图表对比三种策略的拟合效果。")
print("策略 B (标记为 NA) 通常在保留信号和去除噪声之间取得最佳平衡。")
