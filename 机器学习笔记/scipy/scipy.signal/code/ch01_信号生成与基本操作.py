#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第1章 · 信号生成与基本操作 — 配套代码
==============================================
场景定位：金融时间序列 & 电力负荷数据
读者背景：大数据统计专业，有概率论/数理统计基础

演示内容：
  1. 从"频率"视角重新理解你熟悉的金融/电力数据
  2. 标准测试波形生成（chirp, gausspulse, square, sawtooth）
  3. 信号时域特征提取（与描述性统计的对应关系）
  4. 信号可视化最佳实践
  5. 封装 SignalVisualizer 工具类

运行方式：
  python code/ch01_信号生成与基本操作.py

依赖：
  pip install numpy scipy matplotlib
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# ============================================================
# 全局设置
# ============================================================
plt.rcParams.update({
    'figure.dpi': 120,
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
})
print("=" * 60)
print("第1章 · 信号生成与基本操作 — 金融 & 电力场景")
print("=" * 60)

# ============================================================
# Part 1: 用你熟悉的金融数据理解"频率"
# ============================================================
print("\n" + "=" * 60)
print("Part 1: 用金融数据理解'频率'的直觉")
print("=" * 60)

# --- 1.1 构造一个"类股票价格"的信号 ---
# 真实股价 = 长期趋势 + 中期周期 + 短期噪声
# 用信号处理的视角：三组不同频率的正弦波叠加 + 随机噪声

fs_finance = 252          # 年交易日 ≈ 252 天 → "采样率" 252 次/年
trading_days = 4 * 252    # 4 年的交易日 ≈ 1008 天
t_finance = np.arange(trading_days) / fs_finance  # 时间轴，单位：年

np.random.seed(42)

# 成分1：长期趋势 — 极低频率（4年一个上升趋势）
trend = 20 + 15 * t_finance  # 从 20 线性爬到 80

# 成分2：年度周期 — 低频（每年一个涨跌循环，频率 = 1 次/年）
annual_cycle = 8 * np.sin(2 * np.pi * 1.0 * t_finance)

# 成分3：季度周期 — 中等频率（每季度一个涨跌，频率 = 4 次/年）
quarterly_cycle = 4 * np.sin(2 * np.pi * 4.0 * t_finance)

# 成分4：月度周期 — 较高频率（每月一个涨跌，频率 = 12 次/年）
monthly_cycle = 2 * np.sin(2 * np.pi * 12.0 * t_finance)

# 成分5：日度噪声 — 最高频率（每天随机波动）
daily_noise = 1.5 * np.random.randn(trading_days)

# 合成最终"股价"
stock_price = trend + annual_cycle + quarterly_cycle + monthly_cycle + daily_noise

print(f"""
股价合成公式（信号处理视角）：
  股价 = 长期趋势       (≈0 Hz — 整个序列才半个周期)
       + 年度周期       (1 次/年)
       + 季度周期       (4 次/年)
       + 月度周期       (12 次/年)
       + 日度噪声       (高频随机波动)

总样本数: {trading_days}
"采样率": {fs_finance} 次/年
Nyquist 频率: {fs_finance/2:.0f} 次/年 — 最高可检测每年 {fs_finance/2:.0f} 个周期

🎯 关键直觉：
  - 低频成分 = 变化"慢" = 长期趋势、年周期
  - 高频成分 = 变化"快" = 日度涨跌、噪声
  - 滤波 = 选择性地剔除某些频率成分
    比如去掉日度噪声 → 保留趋势+周期 → 你得到一条平滑的"主要趋势线"
""")

# 可视化：信号的频率成分分解
fig, axes = plt.subplots(6, 1, figsize=(14, 10), sharex=True)

components = [
    ('长期趋势 (极低频)', trend, 'tab:blue'),
    ('年度周期 (1次/年)', annual_cycle, 'tab:orange'),
    ('季度周期 (4次/年)', quarterly_cycle, 'tab:green'),
    ('月度周期 (12次/年)', monthly_cycle, 'tab:red'),
    ('日度噪声 (高频)', daily_noise, 'tab:purple'),
    ('合成股价 (全部叠加)', stock_price, 'black'),
]

for ax, (name, data, color) in zip(axes, components):
    ax.plot(t_finance, data, linewidth=0.6, color=color)
    ax.set_ylabel(name.split('(')[0].strip(), fontsize=8)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('时间 (年)')
axes[0].set_title('股价 = 不同"频率"成分的叠加 → 这就是信号处理的视角', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch01_fig1_stock_decomposition.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch01_fig1_stock_decomposition.png")

# ============================================================
# Part 2: 电力负荷数据 — 理解周期和采样
# ============================================================
print("\n" + "=" * 60)
print("Part 2: 电力负荷数据 — 理解采样和混叠")
print("=" * 60)

# 模拟一周的电力负荷数据（每小时一个点）
hours_per_week = 24 * 7
t_hours = np.arange(hours_per_week)  # 时间轴：小时

# 构建一个"类电力负荷"的信号
# 成分1：日周期（24小时）— 白天高、晚上低
daily_pattern = 30 + 15 * np.sin(2 * np.pi * t_hours / 24 - np.pi/2)
# 成分2：半日周期（12小时）— 上午和晚上两个用电高峰
half_day_pattern = 8 * np.sin(2 * np.pi * t_hours / 12 - np.pi/3)
# 成分3：工作日效应 — 工作日高、周末低
weekday_effect = np.where((t_hours // 24) % 7 < 5, 5, -10)  # 0-4=工作日, 5-6=周末
# 成分4：随机波动
np.random.seed(123)
load_noise = 3 * np.random.randn(hours_per_week)

electricity_load = daily_pattern + half_day_pattern + weekday_effect + load_noise

print(f"""
电力负荷合成公式：
  负荷 = 24h 日周期     (频率 = 1/24 次/小时 ≈ 0.042 Hz)
       + 12h 半日周期   (频率 = 1/12 次/小时 ≈ 0.083 Hz)
       + 工作日效应     (频率 = 1/168 次/小时 ≈ 0.006 Hz，即每周一次)
       + 随机波动

样本数: {hours_per_week} 个小时
"采样率": 1 次/小时 — 这决定了我们能检测到的最高频率是每2小时一个周期

🎯 关键直觉：
  - 如果你每小时采一个点，你可以可靠地检测 24h、12h、8h...的周期
  - 但如果你每3小时才采一个点 → Nyquist 频率 = 每6小时一个周期
    → 12h 周期的成分将无法正确被检测（混叠！）
""")

# 可视化：原始 vs 稀疏采样
fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

# 子图1：原始数据（每小时采样）
axes[0].plot(t_hours, electricity_load, linewidth=0.8, color='steelblue')
axes[0].set_ylabel('负荷 (GW)')
axes[0].set_title('原始电力负荷（每小时采样 — fs = 1次/小时）')
axes[0].grid(True, alpha=0.3)

# 子图2：每3小时采样（仍在 Nyquist 极限内，但已经粗糙）
step_3h = 3
t_3h = t_hours[::step_3h]
load_3h = electricity_load[::step_3h]
axes[1].plot(t_hours, electricity_load, linewidth=0.5, alpha=0.3, color='gray', label='原始（参考）')
axes[1].plot(t_3h, load_3h, 'o-', markersize=4, linewidth=1, color='steelblue', label='每3小时采样')
axes[1].set_ylabel('负荷 (GW)')
axes[1].set_title('每3小时采样 — 仍能看到日周期，但细节丢失')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

# 子图3：每25小时采样 — 混叠演示！
# 日周期 = 24h。采样间隔 25h > 24h → Nyquist = 每50h一个周期
# 24h 周期超过 Nyquist → 产生混叠！
step_25h = 25
t_25h = t_hours[::step_25h]
load_25h = electricity_load[::step_25h]
axes[2].plot(t_hours, electricity_load, linewidth=0.5, alpha=0.3, color='gray', label='原始（参考）')
axes[2].plot(t_25h, load_25h, 'o-', markersize=6, linewidth=1.5, color='crimson', label='每25小时采样')
axes[2].set_xlabel('时间 (小时)')
axes[2].set_ylabel('负荷 (GW)')
axes[2].set_title('每25小时采样 → 混叠！日周期被完全扭曲（你看到的是假的慢周期）')
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('code/ch01_fig2_power_load_sampling.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch01_fig2_power_load_sampling.png")

# ============================================================
# Part 3: 标准测试波形 — 信号的"实验材料"
# ============================================================
print("\n" + "=" * 60)
print("Part 3: 标准测试波形生成")
print("=" * 60)

# 准备公用的时间轴
fs = 1000
duration = 1.0
t = np.arange(0, duration, 1/fs)

# --- 3.1 Chirp 扫频信号 ---
print("\n3.1 Chirp — 频率扫动的信号")

# 四种扫频模式的对比
f0, f1 = 5, 100  # 从 5Hz 扫到 100Hz
chirp_linear    = signal.chirp(t, f0, t[-1], f1, method='linear')
chirp_quadratic = signal.chirp(t, f0, t[-1], f1, method='quadratic')
chirp_log       = signal.chirp(t, f0, t[-1], f1, method='logarithmic')
chirp_hyper     = signal.chirp(t, f0, t[-1], f1, method='hyperbolic')

fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True, sharey=True)
chirps = [
    ('linear', chirp_linear, '频率随时间线性增长'),
    ('quadratic', chirp_quadratic, '频率随时间的平方增长（加速扫频）'),
    ('logarithmic', chirp_log, '低频阶段停留更久，高频阶段扫得快'),
    ('hyperbolic', chirp_hyper, '高频阶段停留更久'),
]
for ax, (name, y, desc) in zip(axes, chirps):
    # 只看前500个点，便于观察
    ax.plot(t[:500], y[:500], linewidth=0.5)
    ax.set_ylabel(f'{name}', fontsize=9)
    ax.set_title(f'{name} — {desc}', fontsize=9)
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel('时间 (s)')
fig.suptitle(f'Chirp 信号的四种扫频模式 ({f0}→{f1} Hz)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch01_fig3_chirp_modes.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch01_fig3_chirp_modes.png")

# --- 3.2 高斯脉冲 ---
print("\n3.2 高斯脉冲 — 模拟一次性冲击事件")

fc = 30  # 中心频率 30 Hz
pulse, envelope = signal.gausspulse(t, fc=fc, bw=0.5, retenv=True)

fig, ax = plt.subplots(figsize=(14, 3))
ax.plot(t, pulse, linewidth=0.8, label='高斯脉冲信号 (fc=30Hz)')
ax.plot(t, envelope, '--', linewidth=1.5, color='red', alpha=0.7, label='包络')
ax.plot(t, -envelope, '--', linewidth=1.5, color='red', alpha=0.7)
ax.fill_between(t, -envelope, envelope, alpha=0.08, color='red')
ax.set_xlabel('时间 (s)')
ax.set_title(
    '高斯调制脉冲 — 金融类比：市场受冲击后的震荡衰减',
    fontsize=11, fontweight='bold'
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
plt.tight_layout()
plt.savefig('code/ch01_fig4_gausspulse.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch01_fig4_gausspulse.png")

# --- 3.3 方波、锯齿波 ---
print("\n3.3 方波 & 锯齿波 — 谐波含量最丰富的波形")

f_period = 5  # 5 Hz
t_short = np.linspace(0, 1, fs)

sq_wave = signal.square(2 * np.pi * f_period * t_short, duty=0.5)
sq_wave_narrow = signal.square(2 * np.pi * f_period * t_short, duty=0.2)  # 窄脉冲
saw_wave = signal.sawtooth(2 * np.pi * f_period * t_short, width=1)
saw_wave_rev = signal.sawtooth(2 * np.pi * f_period * t_short, width=0)  # 反向

fig, axes = plt.subplots(2, 2, figsize=(14, 6))
plot_configs = [
    (axes[0, 0], sq_wave, 'tab:blue', '方波 (占空比 50%)'),
    (axes[0, 1], sq_wave_narrow, 'tab:orange', '方波 (占空比 20% — 窄脉冲)'),
    (axes[1, 0], saw_wave, 'tab:green', '锯齿波 (缓慢上升、急剧下降)'),
    (axes[1, 1], saw_wave_rev, 'tab:red', '锯齿波 (急剧上升、缓慢下降)'),
]
for ax, data, color, title in plot_configs:
    ax.plot(t_short[:200], data[:200], linewidth=1, color=color)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel('幅值')
    ax.grid(True, alpha=0.3)
axes[1, 0].set_xlabel('时间 (s)')
axes[1, 1].set_xlabel('时间 (s)')
fig.suptitle('周期波形 — 它们看起来"不光滑"，因为含有丰富的谐波', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch01_fig5_periodic_waves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch01_fig5_periodic_waves.png")

# --- 3.4 单位脉冲 ---
print("\n3.4 单位脉冲 — 信号的'最小单元'")

imp_mid = signal.unit_impulse(50, idx='mid')
imp_first = signal.unit_impulse(50, idx=0)

fig, axes = plt.subplots(1, 2, figsize=(14, 3))
axes[0].stem(np.arange(50), imp_mid, linefmt='C0-', markerfmt='C0o')
axes[0].set_title('δ[n-25] — 时刻 25 的单位冲击')
axes[0].set_xlabel('样本序号 n')
axes[0].grid(True, alpha=0.3)

axes[1].stem(np.arange(50), imp_first, linefmt='C1-', markerfmt='C1o')
axes[1].set_title('δ[n] — 时刻 0 的单位冲击')
axes[1].set_xlabel('样本序号 n')
axes[1].grid(True, alpha=0.3)
fig.suptitle(
    '单位脉冲 δ[n] — 金融类比：某一时刻的一笔"单位冲击"交易',
    fontsize=11, fontweight='bold'
)
plt.tight_layout()
plt.savefig('code/ch01_fig6_unit_impulse.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch01_fig6_unit_impulse.png")

# ============================================================
# Part 4: 时域特征提取 — 和你的统计学知识对接
# ============================================================
print("\n" + "=" * 60)
print("Part 4: 时域特征提取")
print("=" * 60)


def compute_signal_features(x, label=""):
    """
    提取信号的时域基本特征。

    每个特征都在你的统计学知识中有对应：
      均值       ← 样本均值
      标准差     ← 样本标准差
      RMS        ← 二阶原点矩的平方根
      偏度       ← 三阶标准化矩
      峭度       ← 四阶标准化矩
      峰值因子   ← 极值 / RMS（衡量异常尖峰）

    Parameters
    ----------
    x : ndarray (1D)
        信号数据
    label : str
        信号标签

    Returns
    -------
    dict : 特征字典
    """
    N = len(x)
    feat = {'label': label, 'N': N}

    # 一阶、二阶矩
    feat['mean'] = np.mean(x)
    feat['std'] = np.std(x, ddof=0)  # 总体标准差

    # RMS（均方根）— 信号的"有效值"
    feat['rms'] = np.sqrt(np.mean(np.square(x)))

    # 极值
    feat['min'] = np.min(x)
    feat['max'] = np.max(x)
    feat['peak_to_peak'] = feat['max'] - feat['min']

    # 高阶标准化矩（去均值后计算，与统计教材一致）
    x_centered = x - feat['mean']
    sigma = feat['std']
    if sigma > 1e-12:
        feat['skewness'] = np.mean(x_centered ** 3) / (sigma ** 3)
        feat['kurtosis'] = np.mean(x_centered ** 4) / (sigma ** 4)
    else:
        feat['skewness'] = 0.0
        feat['kurtosis'] = 0.0

    # 峰值因子（Crest Factor）— 是否存在异常尖峰
    feat['crest_factor'] = (feat['max'] - feat['min']) / (2 * feat['rms']) if feat['rms'] > 0 else np.inf

    # 过零率 — 信号振荡频率的粗略估计
    zero_crossings = np.sum(np.abs(np.diff(np.sign(x)))) / 2
    feat['zero_crossing_rate'] = zero_crossings / N

    # 能量（信号处理中的"总能量"概念）
    feat['energy'] = np.sum(np.square(x))

    return feat


# --- 构造几个有代表性的信号 ---
np.random.seed(42)
N_points = 2000

# 信号A：平稳的日收益率（白噪声 → 高斯分布，峭度≈3）
returns_normal = np.random.randn(N_points) * 0.02

# 信号B：含"闪崩"的日收益率（尖峰厚尾 → 峭度>>3）
returns_fat_tail = returns_normal.copy()
crash_indices = np.random.choice(N_points, size=5, replace=False)
returns_fat_tail[crash_indices] = np.random.randn(5) * 0.15 - 0.05  # 加入大幅下跌

# 信号C：模拟电网频率（标称50Hz，在49.9-50.1之间波动）
t_freq = np.arange(N_points) / 100  # 假设 100Hz 采样
grid_freq = 50.0 + 0.05 * np.sin(2 * np.pi * 0.1 * t_freq) + 0.02 * np.random.randn(N_points)

# 信号D：模拟分时电价的日间模式（有规律的峰谷）
t_price = np.arange(N_points)
price_pattern = 0.5 + 0.3 * np.sin(2 * np.pi * t_price / 240)   # 日周期
price_pattern += 0.1 * np.sin(2 * np.pi * t_price / 60)          # 更短周期
price_pattern += 0.05 * np.random.randn(N_points)                # 噪声

# --- 横向对比 ---
signals_dict = {
    '日收益率 (近似正态)': returns_normal,
    '日收益率 (含闪崩)':   returns_fat_tail,
    '电网频率 (50Hz附近)':  grid_freq,
    '分时电价模式':           price_pattern,
}

print(f"\n{'信号':<28s} {'N':>6s} {'均值':>8s} {'标准差':>8s} {'偏度':>8s} {'峭度':>8s} {'峰值因子':>8s} {'过零率':>8s}")
print("-" * 90)

for name, sig in signals_dict.items():
    f = compute_signal_features(sig, name)
    print(f"{name:<28s} {f['N']:>6d} {f['mean']:>8.4f} {f['std']:>8.4f} "
          f"{f['skewness']:>8.3f} {f['kurtosis']:>8.3f} {f['crest_factor']:>8.3f} "
          f"{f['zero_crossing_rate']:>8.4f}")

print("""
📊 解读指南（结合你的统计学知识）：
  • "日收益率 (正态)" 的峭度 ≈ 3 — 这正是高斯分布的峰度值
  • "日收益率 (含闪崩)" 的峭度 >> 3 — 尖峰厚尾，极端值比正态分布多
    偏度为负 — 暴跌的幅度大于暴涨的幅度（典型金融数据特征！）
  • "电网频率" 的均值接近 50 — 标称频率；标准差很小 — 频率偏差严格控制
    过零率 ≈ 0.5 — 振荡信号大致每两个采样点穿过一次零线
  • "分时电价" 的峰值因子较高 — 有明显的峰谷差异
    （如果用日内分钟数据会更明显）
""")

# 可视化四个信号的分布形状
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for ax, (name, sig) in zip(axes.flat, signals_dict.items()):
    # 时域波形（前500点）
    ax_inset = ax
    ax_inset.plot(np.arange(min(500, len(sig))), sig[:500], linewidth=0.5)
    feat = compute_signal_features(sig)
    ax_inset.set_title(f'{name}', fontsize=9)
    ax_inset.set_ylabel('值')
    ax_inset.set_xlabel('样本序号')
    ax_inset.grid(True, alpha=0.3)
    # 在图上标注关键统计量
    textstr = (f'偏度={feat["skewness"]:.2f}, '
               f'峭度={feat["kurtosis"]:.2f}, '
               f'峰值因子={feat["crest_factor"]:.2f}')
    ax_inset.text(0.98, 0.97, textstr, transform=ax_inset.transAxes,
                  fontsize=7.5, verticalalignment='top', horizontalalignment='right',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('code/ch01_fig7_signal_features.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch01_fig7_signal_features.png")

# ============================================================
# Part 5: 信号的直方图和分布视角
# ============================================================
print("\n" + "=" * 60)
print("Part 5: 信号 = 随机过程的一次实现 → 直方图 = 边际分布估计")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

for ax, (name, sig) in zip(axes.flat, signals_dict.items()):
    # 绘制直方图（密度估计）+ 叠加正态分布参考线
    ax.hist(sig, bins=60, density=True, alpha=0.6, color='steelblue', edgecolor='white')

    # 叠加拟合的正态分布（用样本均值和标准差）
    mu, std = np.mean(sig), np.std(sig)
    x_pdf = np.linspace(mu - 4*std, mu + 4*std, 200)
    ax.plot(x_pdf, 1/(std*np.sqrt(2*np.pi)) * np.exp(-(x_pdf-mu)**2/(2*std**2)),
            'r-', linewidth=2, label=f'正态分布 N({mu:.3f}, {std:.3f}²)')

    ax.set_title(f'{name} 的幅值分布', fontsize=10)
    ax.legend(fontsize=7)
    ax.set_xlabel('值')
    ax.set_ylabel('概率密度')

fig.suptitle(
    '把信号值看作随机变量的观测 → 直方图 = 边际概率密度的估计',
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
plt.savefig('code/ch01_fig8_signal_histograms.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch01_fig8_signal_histograms.png")
print("""
📊 关键联系：
  你在概率论中学过：
    随机变量 X ~ f(x)
    观测值 x₁, x₂, ..., x_N 是独立同分布样本

  信号处理中的对应：
    信号的 N 个采样点 = 随机过程的 N 次观测
    信号的直方图 → 估计的是边际分布（如果过程是平稳的）
    偏度和峭度 → 描述分布的形状（和统计教材完全一致）

  但要注意：信号不一定是 i.i.d. 的！
    如果信号有自相关（比如今天的股价和昨天相关），
    那么 N 个点并不等价于 N 个独立样本。
    这就是为什么信号处理需要自相关、功率谱等工具（第2、7章）。
""")

# ============================================================
# Part 6: SignalVisualizer 工具类
# ============================================================
print("\n" + "=" * 60)
print("Part 6: SignalVisualizer — 可复用的可视化工具类")
print("=" * 60)


class SignalVisualizer:
    """
    信号可视化工具类。

    设计原则（与统计学图形规范一致）：
      - x 轴始终用物理单位（秒/分钟/天），不用样本序号
      - 标题中始终包含采样率、样本量等元信息
      - 大样本自动适配线宽
      - 金融/电力场景的特殊标注

    使用:
        viz = SignalVisualizer(fs=252)  # 日度金融数据
        viz.plot_waveform(prices, title='Stock Close Price')
    """

    def __init__(self, fs, unit='s'):
        """
        Parameters
        ----------
        fs : float
            采样率 (Hz 或 次/天 等)
        unit : str
            时间轴的单位（如 's', 'day', 'hour'）
        """
        self.fs = fs
        self.unit = unit

    def _make_time_axis(self, signal):
        """自动生成时间轴（单位：self.unit）"""
        return np.arange(len(signal)) / self.fs

    def _auto_linewidth(self, n_points):
        """大信号细线，小信号正常线宽"""
        if n_points > 5000:
            return 0.3
        elif n_points > 1000:
            return 0.6
        return 1.0

    def plot_waveform(self, signal, t=None, ax=None,
                      title='信号波形', color='steelblue',
                      max_points=None):
        """
        绘制时域波形。

        Parameters
        ----------
        signal : ndarray
        t : ndarray or None
            自定义时间轴（None 则自动生成）
        max_points : int or None
            用于长信号的快速预览（只画前 N 个点）
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(14, 3))

        # 截断（用于长信号预览）
        x = signal[:max_points] if max_points else signal

        if t is None:
            t = self._make_time_axis(signal)[:len(x)]
        else:
            t = t[:len(x)]

        lw = self._auto_linewidth(len(x))
        ax.plot(t, x, linewidth=lw, color=color)
        ax.set_xlabel(f'时间 ({self.unit})')
        ax.set_ylabel('值')
        ax.set_title(
            f'{title}  '
            f'(fs={self.fs}次/{self.unit}, '
            f'N={len(signal)}, '
            f'持续={len(signal)/self.fs:.1f}{self.unit})'
        )
        ax.grid(True, alpha=0.3)
        return ax

    def plot_spectrum(self, signal, ax=None, title='幅度谱',
                      max_freq=None, dB=False):
        """
        绘制单边幅度谱（FFT）。

        Parameters
        ----------
        dB : bool
            True → 对数尺度（分贝），False → 线性尺度
        max_freq : float or None
            显示的最高频率上限
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(14, 3))

        N = len(signal)
        X = np.abs(np.fft.rfft(signal)) / N
        X[1:-1] *= 2  # 补偿单边谱
        freqs = np.fft.rfftfreq(N, 1/self.fs)

        # 截断频率范围
        if max_freq is not None:
            mask = freqs <= max_freq
            freqs, X = freqs[mask], X[mask]

        if dB:
            X_display = 20 * np.log10(np.maximum(X, 1e-15))
            ylabel = '幅度 (dB)'
        else:
            X_display = X
            ylabel = '幅度'

        ax.plot(freqs, X_display, linewidth=0.8, color='darkred')
        ax.set_xlabel(f'频率 (次/{self.unit})')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{title}  (Δf={self.fs/N:.3f}次/{self.unit})')
        ax.grid(True, alpha=0.3)
        return ax

    def plot_time_and_frequency(self, signal, t=None,
                                title='时域与频域', max_freq=None):
        """绘制时域+频域的组合视图 — 最常用的综览方式"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6))
        self.plot_waveform(signal, t=t, ax=ax1, title=f'{title} — 时域')
        self.plot_spectrum(signal, ax=ax2,
                           title=f'{title} — 频域', max_freq=max_freq)
        plt.tight_layout()
        return fig


# 使用 SignalVisualizer 展示金融数据
viz = SignalVisualizer(fs=252, unit='年')

# 构造一个带周期+噪声的"类股价"
t_years = np.arange(1008) / 252
demo_price = (100
              + 20 * np.sin(2 * np.pi * 1.0 * t_years)
              + 8 * np.sin(2 * np.pi * 4.0 * t_years)
              + 3 * np.random.randn(1008))

fig = viz.plot_time_and_frequency(
    demo_price,
    title='类股价信号的时域与频域视图',
    max_freq=30  # 只看 30 次/年以下的频率
)
plt.savefig('code/ch01_fig9_time_freq_combo.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch01_fig9_time_freq_combo.png")
print("\n  SignalVisualizer 类已封装完毕，后续章节可直接复用。")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("第1章运行完毕！")
print("=" * 60)
print(f"""
📋 本章产出：

  笔记：doc/01_信号生成与基本操作.md
  代码：code/ch01_信号生成与基本操作.py
  图片：
    ch01_fig1_stock_decomposition.png      — 股价 = 不同频率成分的叠加
    ch01_fig2_power_load_sampling.png      — 电力负荷数据 & 采样混叠
    ch01_fig3_chirp_modes.png              — 四种 chirp 扫频模式
    ch01_fig4_gausspulse.png               — 高斯调制脉冲（冲击事件模拟）
    ch01_fig5_periodic_waves.png           — 方波 & 锯齿波
    ch01_fig6_unit_impulse.png             — 单位脉冲 δ[n]
    ch01_fig7_signal_features.png          — 四种信号的时域特征对比
    ch01_fig8_signal_histograms.png        — 信号 = 随机过程的实现 → 直方图估计边际分布
    ch01_fig9_time_freq_combo.png          — SignalVisualizer 时域+频域综览

🎯 核心收获：
  1. 信号 = 不同频率成分的叠加（就像股价 = 趋势 + 周期 + 噪声）
  2. 频率 = 变化的"快慢"（低频 = 慢变化，高频 = 快变化）
  3. 采样率决定了你能"看到"的最高频率（Nyquist 定理）
  4. 信号的时域特征 = 描述性统计量（均值、方差、偏度、峭度）
  5. scipy.signal 是你的信号处理工具箱

📖 下一站：第2章 — 卷积与相关（移动平均的数学本质）
""")
