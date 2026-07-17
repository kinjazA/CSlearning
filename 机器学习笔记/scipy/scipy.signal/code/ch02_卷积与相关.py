#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第2章 · 卷积与相关 — 配套代码
==============================================
场景定位：金融时间序列 & 电力负荷数据
核心主题：移动平均的数学本质、模板匹配、lead-lag 分析

演示内容：
  1. 卷积 = 滑动加权平均（用股票均线系统直观演示）
  2. mode='full'/'same'/'valid' 的视觉对比
  3. 各种自定义核：SMA / WMA / 三角形 / 差分 / 指数衰减
  4. 相关 = 模板搜索（在价格中找 K 线形态）
  5. 互相关 = lead-lag 分析（两支股票的领先-滞后关系）
  6. 电力负荷平滑 & 突变检测
  7. 性能对比：convolve vs fftconvolve
  8. 封装 ConvolutionAnalyzer 工具类

运行方式：
  python code/ch02_卷积与相关.py

依赖：
  pip install numpy scipy matplotlib pandas
"""

import numpy as np
from scipy import signal
from scipy.signal import convolve, fftconvolve, correlate, convolve2d
import matplotlib.pyplot as plt

# ============================================================
# 全局设置
# ============================================================
plt.rcParams.update({
    'figure.dpi': 120,
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.labelsize': 9,
})
print("=" * 60)
print("第2章 · 卷积与相关 — 金融 & 电力场景")
print("=" * 60)

# ============================================================
# Part 1: 卷积的直觉 — 从均线理解卷积
# ============================================================
print("\n" + "=" * 60)
print("Part 1: 卷积 = 滑动加权平均（用你熟悉的均线来理解）")
print("=" * 60)

# 构造一个"类股价"序列：趋势 + 周期 + 噪声
np.random.seed(42)
N = 200
t = np.arange(N)

# 底层趋势
trend = 50 + 0.1 * t
# 中期周期（模拟市场情绪波动）
cycle = 5 * np.sin(2 * np.pi * t / 40)
# 日度噪声
noise = 2.5 * np.random.randn(N)

price = trend + cycle + noise

print(f"""
构造了一个"类股价"序列（{N} 天）：
  price = 趋势(缓慢上升) + 周期(约40天) + 日度随机噪声

现在用不同卷积核来做平滑，看看效果差异：
""")

# ---- 定义多种卷积核 ----
kernels = {
    'SMA-5\n(5日均线)':           np.ones(5) / 5,
    'SMA-15\n(15日均线)':         np.ones(15) / 15,
    'WMA-5\n(线性加权)':          np.array([1, 2, 3, 4, 5]) / 15,
    'Triangle-9\n(三角形加权)':   np.array([1, 2, 3, 4, 5, 4, 3, 2, 1]) / 25,
    'Exp-15\n(近似指数衰减)':     np.exp(-np.arange(15) / 4),
}

# 对指数核做归一化
kernels['Exp-15\n(近似指数衰减)'] /= kernels['Exp-15\n(近似指数衰减)'].sum()

# ---- 应用所有核 ----
smoothed = {}
for name, kernel in kernels.items():
    smoothed[name] = convolve(price, kernel, mode='same')

# ---- 可视化对比 ----
n_kernels = len(kernels)
fig, axes = plt.subplots(n_kernels + 1, 1, figsize=(16, 12), sharex=True)

# 第一行：原始价格
axes[0].plot(t, price, linewidth=0.6, color='gray', alpha=0.7, label='原始价格')
axes[0].set_ylabel('价格')
axes[0].set_title('原始价格（含趋势 + 周期 + 噪声）', fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=8, loc='upper left')

# 后续行：各种平滑结果
for ax, (name, y) in zip(axes[1:], smoothed.items()):
    ax.plot(t, price, linewidth=0.3, color='gray', alpha=0.4, label='原始（参考）')
    color = 'steelblue' if 'SMA' in name else 'darkorange' if 'WMA' in name else 'green' if 'Tri' in name else 'crimson'
    ax.plot(t, y, linewidth=1.0, color=color, label=name.replace('\n', ' '))
    ax.set_ylabel('价格')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper left')

axes[-1].set_xlabel('交易日')
fig.suptitle('不同卷积核的平滑效果对比 — 均线 = 卷积的一种特例', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch02_fig1_moving_averages.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch02_fig1_moving_averages.png")

# ============================================================
# Part 2: mode 参数 — full vs same vs valid
# ============================================================
print("\n" + "=" * 60)
print("Part 2: mode 参数的三种选择 — 边界怎么处理？")
print("=" * 60)

# 用一个简单信号演示 mode 差异
simple_signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
kernel_sma3 = np.ones(3) / 3

result_full = convolve(simple_signal, kernel_sma3, mode='full')
result_same = convolve(simple_signal, kernel_sma3, mode='same')
result_valid = convolve(simple_signal, kernel_sma3, mode='valid')

# 对齐时间轴用于绘图
n = len(simple_signal)
k = len(kernel_sma3)

# full: 输出从 -(k-1)//2 开始
t_full = np.arange(-(k-1)//2, n + (k//2))

# 可视化
fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=False)

# 原始信号
axes[0].stem(np.arange(n), simple_signal, linefmt='C0-', markerfmt='C0o')
axes[0].set_title(f'原始信号 (长度={n})')
axes[0].set_ylabel('值')
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0, 11)

# mode='full'
colors_full = []
for val in result_full:
    # 标记哪些点只用部分核
    idx = list(result_full).index(val) if val == result_full[0] else 0
colors_full = ['dimgray' if i < (k-1)//2 or i >= n + (k-1)//2 - (k//2) else 'steelblue'
               for i in range(len(result_full))]
markerline, stemlines, baseline = axes[1].stem(t_full, result_full, linefmt='C1-', markerfmt='C1o')
axes[1].set_title(f"mode='full' → 输出长度 = {len(result_full)} "
                  f"(灰色 = 部分核覆盖的点, 边界不准确)")
axes[1].set_ylabel('值')
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0, 11)

# mode='same'
t_same = np.arange(n)
colors_same = ['dimgray' if i < (k-1)//2 or i >= n - (k//2) else 'steelblue'
               for i in range(n)]
axes[2].stem(t_same, result_same, linefmt='C2-', markerfmt='C2o')
axes[2].set_title(f"mode='same' → 输出长度 = {len(result_same)} "
                  f"(与原始对齐！灰色 = 边界伪影)")
axes[2].set_ylabel('值')
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim(0, 11)

# mode='valid'
t_valid = np.arange((k-1)//2, n - (k//2))
axes[3].stem(t_valid, result_valid, linefmt='C3-', markerfmt='C3o')
axes[3].set_title(f"mode='valid' → 输出长度 = {len(result_valid)} "
                  f"(全部准确，但序列缩短，时间轴偏移)")
axes[3].set_xlabel('样本序号')
axes[3].set_ylabel('值')
axes[3].grid(True, alpha=0.3)
axes[3].set_ylim(0, 11)

fig.suptitle('卷积 mode 参数对比 — SMA-3 核 [1/3, 1/3, 1/3]', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch02_fig2_convolution_modes.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch02_fig2_convolution_modes.png")
print("""
  实战建议：
  - 金融均线 → mode='same'（对齐时间轴），但前 (M-1)//2 个点手动标为不可靠
  - 特征提取（不需要对齐）→ mode='valid'（干净无边界问题）
  - 检查边界效应 → mode='full'（看到全部情况）
""")

# ============================================================
# Part 3: 自定义核的威力 — 差分、边缘检测、EMA
# ============================================================
print("\n" + "=" * 60)
print("Part 3: 自定义核 — 不只是平滑，可以做更多")
print("=" * 60)

# 复用 Part 1 的价格数据
price = trend + cycle + noise

# ---- 核1：一阶差分 → 日涨跌 ----
diff1_kernel = np.array([-1, 1])
price_diff1 = convolve(price, diff1_kernel, mode='same')

# ---- 核2：二阶差分 → 涨跌的加速度（趋势是否在衰竭）----
diff2_kernel = np.array([1, -2, 1])
price_diff2 = convolve(price, diff2_kernel, mode='same')

# ---- 核3：中心差分 → 更平滑的一阶差分（避免偏移）----
center_diff_kernel = np.array([-1, 0, 1]) / 2
price_center_diff = convolve(price, center_diff_kernel, mode='same')

# ---- 核4：[-1,-1,-1, 0, 1,1,1] → 趋势强度指标 ----
trend_kernel = np.array([-1, -1, -1, 0, 1, 1, 1]) / 3
price_trend_strength = convolve(price, trend_kernel, mode='same')

print("""
自定义卷积核一览：
  [-1, 1]        → 一阶差分（相当于 pandas .diff(1)）
  [1, -2, 1]     → 二阶差分（衡量"变化的变化"）
  [-1, 0, 1]/2   → 中心差分（比简单差分更平滑）
  [-1,-1,-1,0,1,1,1]/3 → 趋势强度（未来3天 vs 过去3天）
""")

# 可视化
fig, axes = plt.subplots(5, 1, figsize=(16, 10), sharex=True)

axes[0].plot(t, price, linewidth=0.7, color='gray')
axes[0].set_ylabel('价格')
axes[0].set_title('原始价格')
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, price_diff1, linewidth=0.7, color='steelblue')
axes[1].axhline(y=0, color='black', linewidth=0.5, linestyle='--')
axes[1].set_ylabel('Δ价格')
axes[1].set_title('一阶差分 [-1, 1] → 每日涨跌')
axes[1].grid(True, alpha=0.3)

axes[2].plot(t, price_diff2, linewidth=0.7, color='darkorange')
axes[2].axhline(y=0, color='black', linewidth=0.5, linestyle='--')
axes[2].set_ylabel('Δ²价格')
axes[2].set_title('二阶差分 [1, -2, 1] → 涨跌的加速度（正值=上涨加速，负值=上涨减速）')
axes[2].grid(True, alpha=0.3)

axes[3].plot(t, price_center_diff, linewidth=0.7, color='green')
axes[3].axhline(y=0, color='black', linewidth=0.5, linestyle='--')
axes[3].set_ylabel('中心差分')
axes[3].set_title('中心差分 [-1, 0, 1]/2 → 左右对称，更准确地定位变化时刻')
axes[3].grid(True, alpha=0.3)

axes[4].plot(t, price_trend_strength, linewidth=0.7, color='crimson')
axes[4].axhline(y=0, color='black', linewidth=0.5, linestyle='--')
axes[4].fill_between(t, 0, price_trend_strength,
                     where=(price_trend_strength > 0), alpha=0.15, color='green')
axes[4].fill_between(t, 0, price_trend_strength,
                     where=(price_trend_strength < 0), alpha=0.15, color='red')
axes[4].set_xlabel('交易日')
axes[4].set_ylabel('趋势强度')
axes[4].set_title('趋势强度（未来3天均值 - 过去3天均值）→ 绿=上升趋势，红=下降趋势')
axes[4].grid(True, alpha=0.3)

fig.suptitle('卷积不只是平滑 — 差分核让你提取各种金融信号特征', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch02_fig3_custom_kernels.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch02_fig3_custom_kernels.png")

# ============================================================
# Part 4: 相关 — 模板匹配 & lead-lag 分析
# ============================================================
print("\n" + "=" * 60)
print("Part 4: 相关 — 在信号中搜索模式 & 判断谁领先谁")
print("=" * 60)

# ---- 4.1 模板匹配：在价格序列中搜索"W底"形态 ----
print("\n4.1 模板匹配 — 搜索 W 底形态")

# 构造一个"W 底"形态模板
w_bottom = np.array([-1.5, -1.0, -0.5, -0.2, -0.8, -1.3, -0.5, 0.0, 0.5, 1.0])
w_bottom = w_bottom - np.mean(w_bottom)  # 去均值（只匹配形状，不匹配绝对水平）

# 构造一个包含 W 底形态的价格序列
np.random.seed(123)
price_long = np.random.randn(300) * 1.0 + np.sin(np.linspace(0, 4*np.pi, 300)) * 2

# 在位置 80 和 200 嵌入 W 底形态
price_long[80:90] += w_bottom * 1.5
price_long[200:210] += w_bottom * 1.2

# 用相关搜索
match_scores = correlate(price_long, w_bottom, mode='same')

print(f"""
  模板长度 = {len(w_bottom)} 个点
  在位置 80-89 和 200-209 嵌入了两个"W底"形态
  用 correlate() 在整个序列中搜索...
""")

# 找到匹配度最高的几个位置
top_k = 5
# 排除边界（模板长度一半的范围）
valid_start = len(w_bottom) // 2
valid_end = len(match_scores) - len(w_bottom) // 2
match_scores_masked = match_scores.copy()
match_scores_masked[:valid_start] = -np.inf
match_scores_masked[valid_end:] = -np.inf
top_indices = np.argsort(match_scores_masked)[-top_k:][::-1]

print(f"  匹配得分最高的 {top_k} 个位置：{top_indices}")
print(f"  实际嵌入位置：80 和 200")
print(f"  → {'' if any(abs(idx - 80) <= 3 or abs(idx - 200) <= 3 for idx in top_indices) else '未'}成功检测到！")

# 可视化
fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)

axes[0].plot(price_long, linewidth=0.5, color='gray')
axes[0].set_ylabel('价格')
axes[0].set_title('包含两个"W底"形态的价格序列')
axes[0].grid(True, alpha=0.3)
# 标记嵌入位置
for pos, color in [(80, 'red'), (200, 'red')]:
    axes[0].axvspan(pos, pos+10, alpha=0.2, color=color)
axes[0].text(85, axes[0].get_ylim()[1]*0.9, 'W底①', fontsize=8, color='red', ha='center')
axes[0].text(205, axes[0].get_ylim()[1]*0.9, 'W底②', fontsize=8, color='red', ha='center')

axes[1].plot(np.arange(len(w_bottom)), w_bottom, 'o-', linewidth=2, markersize=5, color='darkorange')
axes[1].set_ylabel('模板值')
axes[1].set_title(f'W底形态模板 (去均值后, 长度={len(w_bottom)})')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0, color='black', linewidth=0.5, linestyle='--')

axes[2].plot(match_scores, linewidth=0.7, color='steelblue')
axes[2].set_ylabel('匹配得分')
axes[2].set_xlabel('位置')
axes[2].set_title('correlate() 输出 → 两个明显的峰值对应两个 W 底')
axes[2].grid(True, alpha=0.3)
# 标记检测到的峰值
for idx in top_indices:
    axes[2].axvline(x=idx, color='red', linewidth=1.5, alpha=0.7, linestyle='--')

plt.tight_layout()
plt.savefig('code/ch02_fig4_template_matching.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch02_fig4_template_matching.png")

# ---- 4.2 Lead-Lag 分析 — 两支股票的领先/滞后关系 ----
print("\n4.2 Lead-Lag 分析 — 谁领先谁？")

# 模拟两支行业相近的股票日收益率
np.random.seed(42)
N_stocks = 500

# 公共因子（行业共同驱动）
common_factor = np.random.randn(N_stocks) * 0.01

# 股票 A："龙头股" — 首先反映信息
returns_A = common_factor + np.random.randn(N_stocks) * 0.008

# 股票 B：滞后 2 天的"跟随股" — 对同一信息反应较慢
lag_days = 2
returns_B = np.zeros(N_stocks)
returns_B[lag_days:] = common_factor[:-lag_days] * 0.8 + np.random.randn(N_stocks - lag_days) * 0.01
returns_B[:lag_days] = np.random.randn(lag_days) * 0.01

# 计算互相关
cross_corr = correlate(returns_A - np.mean(returns_A),
                       returns_B - np.mean(returns_B),
                       mode='full')
lags = np.arange(-N_stocks + 1, N_stocks)

# 找到互相关最大的 lag
best_lag = lags[np.argmax(np.abs(cross_corr))]
print(f"""
  真实滞后：B 滞后 A {lag_days} 天
  互相关检测到的滞后：{best_lag} 天
  → {'✅ 准确检测！' if best_lag == lag_days else '⚠️ 有偏差'}

  解读：
  - 如果互相关最大值出现在正 lag → B 滞后于 A（A 领先）
  - 如果互相关最大值出现在负 lag → A 滞后于 B（B 领先）
  - lag=0 峰值 → 二者同步
""")

# 可视化
fig, axes = plt.subplots(3, 1, figsize=(16, 8))

# 收益率序列（前100天）
axes[0].plot(returns_A[:100], linewidth=0.8, label='股票 A (龙头)', color='steelblue')
axes[0].plot(returns_B[:100], linewidth=0.8, label='股票 B (跟随)', color='darkorange', alpha=0.8)
axes[0].set_ylabel('日收益率')
axes[0].set_title('两支股票的日收益率（前100天）')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# 标记一个共同的波动，观察 B 滞后 A
zoom_start, zoom_end = 30, 60
axes[1].plot(np.arange(zoom_start, zoom_end), returns_A[zoom_start:zoom_end],
             'o-', linewidth=1.5, markersize=3, label='A (龙头)', color='steelblue')
axes[1].plot(np.arange(zoom_start, zoom_end), returns_B[zoom_start:zoom_end],
             's-', linewidth=1.5, markersize=3, label='B (跟随)', color='darkorange')
axes[1].set_ylabel('日收益率')
axes[1].set_title(f'放大第{zoom_start}-{zoom_end}天 → 观察 B 的反应是否滞后于 A')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

# 互相关函数
# 只看 ±20 的 lag 范围
plot_lag_range = 20
mask = np.abs(lags) <= plot_lag_range
axes[2].plot(lags[mask], cross_corr[mask], linewidth=1.0, color='steelblue')
axes[2].axvline(x=best_lag, color='red', linewidth=1.5, linestyle='--',
                label=f'最大相关 lag={best_lag}')
axes[2].axvline(x=0, color='gray', linewidth=0.5, linestyle='-')
axes[2].set_xlabel('Lag (天) → 正= B 滞后于 A, 负= A 滞后于 B')
axes[2].set_ylabel('互相关')
axes[2].set_title('互相关函数 → 峰值位置 = B 相对于 A 的滞后天数')
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('code/ch02_fig5_lead_lag.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch02_fig5_lead_lag.png")

# ============================================================
# Part 5: 电力负荷场景 — 平滑 & 突变检测
# ============================================================
print("\n" + "=" * 60)
print("Part 5: 电力负荷场景 — 日负荷曲线平滑 & 负荷突变检测")
print("=" * 60)

# 模拟一周的电力负荷数据（15分钟一个点 = 每天96个点）
np.random.seed(99)
points_per_day = 96
days = 7
N_load = points_per_day * days
t_load = np.arange(N_load) / points_per_day  # 单位：天

# 日负荷模式：凌晨低、上午爬升、下午高峰、晚上下降
one_day = np.zeros(points_per_day)
# 凌晨低谷 (0:00-6:00)
one_day[0:24] = 0.3
# 上午爬升 (6:00-9:00)
one_day[24:36] = np.linspace(0.3, 0.7, 12)
# 上午工作高峰 (9:00-12:00)
one_day[36:48] = 0.7 + 0.1 * np.sin(np.linspace(0, np.pi, 12))
# 中午稍降 (12:00-14:00)
one_day[48:56] = np.linspace(0.8, 0.6, 8)
# 下午第二个高峰 (14:00-17:00)
one_day[56:68] = 0.6 + 0.2 * np.sin(np.linspace(0, np.pi/2, 12))
# 晚间下降 (17:00-22:00)
one_day[68:88] = np.linspace(0.8, 0.4, 20)
# 夜间 (22:00-24:00)
one_day[88:96] = np.linspace(0.4, 0.3, 8)

# 工作日/周末差异
load = np.tile(one_day, days)
# 周末降低
load[5*points_per_day:6*points_per_day] *= 0.8
load[6*points_per_day:7*points_per_day] *= 0.75

# 加入噪声
load_noisy = load + 0.05 * np.random.randn(N_load)

# 在某个位置加入"电弧炉启动"的冲击（模拟大型工业设备）
furnace_start = 3 * points_per_day + 30  # 第3天第30个点
load_noisy[furnace_start:furnace_start+8] += 0.4 * np.exp(-np.arange(8)/3)

print(f"""
模拟了 7 天的电力负荷数据（每15分钟一个点，共 {N_load} 个点）：
  - 正常的日周期模式（早峰 + 晚峰）
  - 工作日/周末差异
  - 随机噪声
  - 第3天插入了一个"电弧炉启动"的负荷冲击

现在演示：
  1. 用卷积平滑 → 提取干净的日负荷曲线
  2. 用差分核 → 自动检测负荷突变
""")

# ---- 平滑 ----
# 三角形核（约1.5小时窗口）
kernel_size = 7
tri_kernel = signal.windows.triang(kernel_size)
tri_kernel = tri_kernel / tri_kernel.sum()
load_smoothed = convolve(load_noisy, tri_kernel, mode='same')

# ---- 突变检测 ----
# 核：未来5个点的平均 减去 过去5个点的平均
edge_kernel = np.ones(11)
edge_kernel[5:] = 1   # 未来
edge_kernel[:5] = -1  # 过去
edge_kernel[5] = 0    # 当前点不加权
edge_kernel = edge_kernel / 5
load_edge = convolve(load_noisy, edge_kernel, mode='same')

# 找到突变点（负荷突然上升超过阈值）
threshold = 0.15
spikes = np.where(load_edge > threshold)[0]

print(f"  检测到 {len(spikes)} 个突变点超过阈值 {threshold}")
print(f"  其中电弧炉启动位置 = {furnace_start}")
near_furnace = [s for s in spikes if abs(s - furnace_start) <= 5]
print(f"  在电弧炉位置附近(±5)检测到的突变点：{near_furnace}")

# 可视化
fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)

# 原始 vs 平滑
axes[0].plot(t_load, load_noisy, linewidth=0.3, color='gray', alpha=0.5, label='原始（含噪声）')
axes[0].plot(t_load, load_smoothed, linewidth=1.0, color='steelblue', label='三角形核平滑')
axes[0].set_ylabel('负荷 (归一化)')
axes[0].set_title('电力负荷 — 原始数据 vs 卷积平滑')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# 原始 vs 无噪 ground truth
axes[1].plot(t_load, load, linewidth=1.0, color='black', alpha=0.7, label='真实负荷模式 (ground truth)')
axes[1].plot(t_load, load_smoothed, linewidth=1.0, color='steelblue', alpha=0.8, label='卷积平滑后')
axes[1].set_ylabel('负荷 (归一化)')
axes[1].set_title('卷积平滑 vs 真实模式 → 卷积成功提取了干净的日负荷形状')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

# 突变检测
axes[2].plot(t_load, load_edge, linewidth=0.7, color='crimson')
axes[2].axhline(y=threshold, color='red', linewidth=1, linestyle='--', label=f'阈值 = {threshold}')
axes[2].axhline(y=0, color='black', linewidth=0.5)
# 标记电弧炉位置
axes[2].axvline(x=furnace_start / points_per_day, color='orange', linewidth=2,
                alpha=0.7, linestyle='--', label=f'电弧炉启动 (点{furnace_start})')
axes[2].set_xlabel('时间 (天)')
axes[2].set_ylabel('突变强度')
axes[2].set_title('负荷突变检测 — 在电弧炉启动位置出现了明显尖峰 ✅')
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

fig.suptitle('电力负荷分析 — 卷积平滑 + 突变检测', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch02_fig6_power_load_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch02_fig6_power_load_analysis.png")

# ============================================================
# Part 6: 性能对比 — convolve vs fftconvolve
# ============================================================
print("\n" + "=" * 60)
print("Part 6: 性能对比 — 什么时候用 fftconvolve？")
print("=" * 60)

import time

# 测试不同核长度下的性能
signal_len = 10000
x = np.random.randn(signal_len)

kernel_lengths = [5, 20, 50, 100, 200, 500, 1000, 2000, 5000]
times_direct = []
times_fft = []

for M in kernel_lengths:
    h = np.ones(M) / M

    # 直接卷积
    t0 = time.perf_counter()
    for _ in range(10):
        convolve(x, h, mode='same')
    t_direct = (time.perf_counter() - t0) / 10
    times_direct.append(t_direct * 1000)  # 转为 ms

    # FFT 卷积
    t0 = time.perf_counter()
    for _ in range(10):
        fftconvolve(x, h, mode='same')
    t_fft = (time.perf_counter() - t0) / 10
    times_fft.append(t_fft * 1000)

print(f"\n  信号长度 N = {signal_len}")
print(f"  {'核长度 M':<12s} {'直接卷积 (ms)':<16s} {'FFT卷积 (ms)':<16s} {'推荐':<10s}")
print(f"  {'-'*50}")
for i, M in enumerate(kernel_lengths):
    faster = '直接' if times_direct[i] < times_fft[i] else 'FFT'
    print(f"  {M:<12d} {times_direct[i]:<16.4f} {times_fft[i]:<16.4f} {faster:<10s}")

# 可视化
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(kernel_lengths, times_direct, 'o-', linewidth=2, markersize=6, label='convolve (直接)', color='steelblue')
ax.plot(kernel_lengths, times_fft, 's-', linewidth=2, markersize=6, label='fftconvolve (FFT)', color='darkorange')
ax.set_xlabel('核长度 M')
ax.set_ylabel('耗时 (ms)')
ax.set_title(f'卷积性能对比 — 信号长度 N={signal_len}')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 找到交叉点
for i in range(len(kernel_lengths)-1):
    if (times_direct[i] < times_fft[i]) and (times_direct[i+1] > times_fft[i+1]):
        cross_M = kernel_lengths[i+1]
        ax.axvline(x=cross_M, color='red', linewidth=1, linestyle='--',
                   alpha=0.7, label=f'交叉点 ≈ M={cross_M}')
        ax.legend(fontsize=10)
        break

plt.tight_layout()
plt.savefig('code/ch02_fig7_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch02_fig7_performance.png")

# ============================================================
# Part 7: ConvolutionAnalyzer 工具类
# ============================================================
print("\n" + "=" * 60)
print("Part 7: ConvolutionAnalyzer — 可复用的分析工具类")
print("=" * 60)


class ConvolutionAnalyzer:
    """
    卷积分析工具类 — 封装常用的金融/电力信号卷积操作。

    使用：
        ca = ConvolutionAnalyzer()
        sma = ca.sma(prices, window=5)
        diff = ca.diff(prices, order=1)
        strength = ca.trend_strength(prices, lookback=3)
    """

    @staticmethod
    def moving_average(x, window, kernel_type='uniform'):
        """
        各种均线的统一接口。

        Parameters
        ----------
        x : ndarray
            输入信号
        window : int
            窗口长度（必须为奇数，方便对称）
        kernel_type : str
            'uniform' / 'triangle' / 'linear_up' / 'linear_down'

        Returns
        -------
        ndarray (same length as x via mode='same')
        """
        if window % 2 == 0:
            window += 1  # 保证奇数，对称

        if kernel_type == 'uniform':
            kernel = np.ones(window)
        elif kernel_type == 'triangle':
            kernel = signal.windows.triang(window)
        elif kernel_type == 'linear_up':
            kernel = np.arange(1, window + 1)  # 越近权重越大
        elif kernel_type == 'linear_down':
            kernel = np.arange(window, 0, -1)  # 越远的权重越大
        else:
            raise ValueError(f"Unknown kernel_type: {kernel_type}")

        kernel = kernel / kernel.sum()
        return convolve(x, kernel, mode='same')

    @staticmethod
    def sma(x, window):
        """简单移动平均"""
        return ConvolutionAnalyzer.moving_average(x, window, 'uniform')

    @staticmethod
    def diff(x, order=1):
        """
        差分（价格变化）。

        Parameters
        ----------
        order : int
            1 = 一阶差分 [-1, 1]
            2 = 二阶差分 [1, -2, 1]
        """
        if order == 1:
            return convolve(x, np.array([-1, 1]), mode='same')
        elif order == 2:
            return convolve(x, np.array([1, -2, 1]), mode='same')

    @staticmethod
    def trend_strength(x, lookback=3):
        """
        趋势强度 = 未来 N 天均值 - 过去 N 天均值。
        正值 → 上升趋势，负值 → 下降趋势。
        """
        n = lookback
        kernel = np.ones(2 * n + 1)
        kernel[:n] = -1
        kernel[n] = 0
        kernel[n+1:] = 1
        kernel = kernel / n
        return convolve(x, kernel, mode='same')

    @staticmethod
    def find_pattern(signal_data, pattern):
        """
        在信号中搜索模板匹配。

        Returns
        -------
        scores : ndarray
            每个位置的匹配得分（越高峰 = 越匹配）
        """
        # 去均值，只匹配形状
        pattern_centered = pattern - np.mean(pattern)
        return correlate(signal_data, pattern_centered, mode='same')

    @staticmethod
    def lead_lag(x, y, max_lag=None):
        """
        分析两个信号的领先-滞后关系。

        Parameters
        ----------
        x, y : ndarray
            两个信号（建议先做去均值和标准化）
        max_lag : int or None
            搜索的最大滞后范围

        Returns
        -------
        best_lag : int
            互相关最大时的 lag（正数 = y 滞后于 x）
        lags : ndarray
            所有 lag 值
        corr : ndarray
            所有 lag 对应的互相关值
        """
        N = len(x)
        corr = correlate(x - np.mean(x), y - np.mean(y), mode='full')
        lags = np.arange(-N + 1, N)

        if max_lag is not None:
            mask = np.abs(lags) <= max_lag
            lags, corr = lags[mask], corr[mask]

        best_lag = lags[np.argmax(np.abs(corr))]
        return best_lag, lags, corr


# 演示 ConvolutionAnalyzer
ca = ConvolutionAnalyzer()

# 用 Part 1 的价格数据
sma5 = ca.sma(price, 5)
sma15 = ca.sma(price, 15)
triangle15 = ca.moving_average(price, 15, 'triangle')
trend = ca.trend_strength(price, lookback=3)

fig, axes = plt.subplots(3, 1, figsize=(16, 7), sharex=True)

axes[0].plot(t, price, linewidth=0.4, color='gray', alpha=0.5, label='原始')
axes[0].plot(t, sma5, linewidth=1, label='SMA-5')
axes[0].plot(t, sma15, linewidth=1, label='SMA-15')
axes[0].plot(t, triangle15, linewidth=1, label='Triangle-15')
axes[0].set_ylabel('价格')
axes[0].set_title('ConvolutionAnalyzer — 均线系统')
axes[0].legend(fontsize=8, ncol=4)
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, ca.diff(price, order=1), linewidth=0.7, color='steelblue')
axes[1].axhline(y=0, color='black', linewidth=0.5, linestyle='--')
axes[1].set_ylabel('Δ价格')
axes[1].set_title('一阶差分')
axes[1].grid(True, alpha=0.3)

axes[2].plot(t, trend, linewidth=0.7, color='crimson')
axes[2].axhline(y=0, color='black', linewidth=0.5, linestyle='--')
axes[2].fill_between(t, 0, trend, where=(trend>0), alpha=0.15, color='green')
axes[2].fill_between(t, 0, trend, where=(trend<0), alpha=0.15, color='red')
axes[2].set_xlabel('交易日')
axes[2].set_ylabel('趋势强度')
axes[2].set_title('趋势强度 (未来3天 vs 过去3天)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('code/ch02_fig8_analyzer_demo.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch02_fig8_analyzer_demo.png")
print("  ConvolutionAnalyzer 类已封装完毕，后续章节可直接复用。")


# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("第2章运行完毕！")
print("=" * 60)
print(f"""
📋 本章产出：

  笔记：doc/02_卷积与相关.md
  代码：code/ch02_卷积与相关.py
  图片：
    ch02_fig1_moving_averages.png      — 各种均线核的平滑效果对比
    ch02_fig2_convolution_modes.png    — full/same/valid 边界处理详解
    ch02_fig3_custom_kernels.png       — 差分核、趋势强度等自定义卷积
    ch02_fig4_template_matching.png    — 用相关在价格序列中搜索 W 底形态
    ch02_fig5_lead_lag.png             — 互相关检测股票领先-滞后关系
    ch02_fig6_power_load_analysis.png  — 电力负荷平滑 + 突变检测
    ch02_fig7_performance.png          — convolve vs fftconvolve 性能对比
    ch02_fig8_analyzer_demo.png        — ConvolutionAnalyzer 工具类演示

🎯 核心收获：
  1. 卷积 = 滑动加权平均 → 你熟悉的均线就是卷积
  2. 不同的核 = 不同的"看数据的方式"
     - 均匀核 → 简单平滑
     - 三角形核 → 更平滑（加权中心）
     - 差分核 → 检测变化
     - 自定义核 → 任何线性特征提取
  3. 相关 = 模板匹配 → 在序列中"搜索"特定模式
  4. 互相关 = lead-lag 分析 → 判断两个序列谁领先谁
  5. mode='same' 最常用但要处理边界伪影

📖 下一站：第3章 — FIR 滤波器设计（上）窗口法
  → 卷积是滤波器的数学基础，第3章开始设计"专业级"的滤波器
""")
