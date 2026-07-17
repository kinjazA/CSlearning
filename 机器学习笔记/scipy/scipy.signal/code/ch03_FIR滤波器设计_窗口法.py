#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第3章 · FIR 滤波器设计（上）— 窗口法 — 配套代码
=====================================================
场景定位：金融时间序列 & 电力负荷数据
核心主题：用窗函数法设计 FIR 滤波器，并进行频率响应分析

演示内容：
  1. 窗函数可视化对比（boxcar / hann / hamming / blackman / kaiser）
  2. Kaiser 窗 β 参数扫参实验
  3. Gibbs 现象演示
  4. firwin 设计四种滤波器（低通/高通/带通/带阻）
  5. freqz 频率响应分析 & group_delay 群延迟
  6. 金融实战：提取股价中期趋势（低通）、检测特定周期（带通）
  7. 电力实战：提取日负荷周期（带通）、去除谐波（带阻）
  8. 封装 FIRFilterDesigner 工具类

运行方式：
  python code/ch03_FIR滤波器设计_窗口法.py

依赖：
  pip install numpy scipy matplotlib
"""

import numpy as np
from scipy import signal
from scipy.signal import firwin, firwin2, freqz, group_delay, lfilter, filtfilt
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
print("第3章 · FIR 滤波器设计（上）— 窗口法")
print("=" * 60)

# ============================================================
# Part 1: 窗函数对比 — 形状与频谱
# ============================================================
print("\n" + "=" * 60)
print("Part 1: 窗函数对比 — 形状决定频谱")
print("=" * 60)

M = 51  # 窗的长度
n = np.arange(M)

windows = {
    'boxcar (矩形)':      signal.windows.boxcar(M),
    'hann':              signal.windows.hann(M),
    'hamming':           signal.windows.hamming(M),
    'blackman':          signal.windows.blackman(M),
    'kaiser (β=5)':      signal.windows.kaiser(M, beta=5),
    'kaiser (β=8.6)':    signal.windows.kaiser(M, beta=8.6),
    'kaiser (β=12)':     signal.windows.kaiser(M, beta=12),
}

# 可视化：时域形状 + 频域幅度响应
fig, axes = plt.subplots(len(windows), 2, figsize=(14, 14))

for i, (name, win) in enumerate(windows.items()):
    # 左列：时域形状
    axes[i, 0].plot(n, win, 'o-', linewidth=1.5, markersize=3)
    axes[i, 0].set_ylabel(name, fontsize=9, rotation=0, labelpad=60)
    axes[i, 0].set_ylim(-0.05, 1.15)
    axes[i, 0].grid(True, alpha=0.3)
    if i == 0:
        axes[i, 0].set_title('时域形状 (窗函数权重)', fontsize=9)

    # 右列：频域幅度响应（dB）
    # 加零填充以获得更平滑的频率响应
    W = np.abs(np.fft.rfft(win, n=2048))
    W_db = 20 * np.log10(np.maximum(W / W.max(), 1e-15))
    freqs = np.linspace(0, 0.5, len(W_db))
    axes[i, 1].plot(freqs, W_db, linewidth=0.8)
    axes[i, 1].set_ylabel(name, fontsize=9, rotation=0, labelpad=60)
    axes[i, 1].set_ylim(-120, 5)
    axes[i, 1].grid(True, alpha=0.3)
    if i == 0:
        axes[i, 1].set_title('频域响应 (dB, 归一化频率)', fontsize=9)

    # 标注旁瓣水平（略过 boxcar 因为旁瓣太明显）
    if 'boxcar' not in name:
        # 找第一旁瓣峰值
        W_slice = W_db[10:]  # 跳过主瓣
        first_lobe = np.max(W_slice[:100])
        axes[i, 1].axhline(y=first_lobe, color='red', linewidth=0.5,
                           linestyle='--', alpha=0.5)
        axes[i, 1].text(0.35, first_lobe + 3, f'{first_lobe:.0f} dB',
                        fontsize=7, color='red')

axes[-1, 0].set_xlabel('样本序号')
axes[-1, 1].set_xlabel('归一化频率 (× π rad/sample)')

fig.suptitle('窗函数对比 — 旁瓣越低 = 过渡带越宽（没有免费的午餐）',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch03_fig1_window_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch03_fig1_window_comparison.png")

# ============================================================
# Part 2: Kaiser 窗 β 参数扫参
# ============================================================
print("\n" + "=" * 60)
print("Part 2: Kaiser 窗 — β 参数决定一切")
print("=" * 60)

betas = [0, 2, 4, 5, 6, 8, 10, 12]
M_kaiser = 51
t_kaiser = np.arange(M_kaiser)

fig, axes = plt.subplots(2, 1, figsize=(14, 7))

colors = plt.cm.viridis(np.linspace(0, 1, len(betas)))

for beta, color in zip(betas, colors):
    win = signal.windows.kaiser(M_kaiser, beta=beta)

    # 时域
    axes[0].plot(t_kaiser, win, linewidth=1.2, color=color,
                 label=f'β={beta}', alpha=0.85)

    # 频域
    W = np.abs(np.fft.rfft(win, n=2048))
    W_db = 20 * np.log10(np.maximum(W / W.max(), 1e-15))
    freqs = np.linspace(0, 0.5, len(W_db))
    axes[1].plot(freqs, W_db, linewidth=1.0, color=color,
                 label=f'β={beta}', alpha=0.85)

axes[0].set_xlabel('样本序号')
axes[0].set_ylabel('权重')
axes[0].set_title('Kaiser 窗时域形状 — β越大，两端衰减越快')
axes[0].legend(fontsize=7, ncol=4)
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('归一化频率 (× π rad/sample)')
axes[1].set_ylabel('幅度 (dB)')
axes[1].set_title('Kaiser 窗频域响应 — β越大，旁瓣越低，主瓣越宽')
axes[1].set_ylim(-140, 5)
axes[1].legend(fontsize=7, ncol=4)
axes[1].grid(True, alpha=0.3)

fig.suptitle('Kaiser 窗 β 参数扫参 — 一窗在手，天下我有',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch03_fig2_kaiser_beta_sweep.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch03_fig2_kaiser_beta_sweep.png")
print("""
  β 选择指南：
    β=0   → 矩形窗 (旁瓣 -13dB)     几乎不用
    β=3~5 → 类似于 hann (-30dB)     一般平滑
    β=5~7 → 类似于 hamming (-40dB)  常用推荐
    β=8~10 → 类似于 blackman (-60dB) 高衰减需求
    β=12+ → 极端衰减 (-90dB+)       精确频率分离
""")

# ============================================================
# Part 3: Gibbs 现象 — 为什么不能直接截断？
# ============================================================
print("\n" + "=" * 60)
print("Part 3: Gibbs 现象 — 直接截断 = 频域振荡")
print("=" * 60)

# 构造理想低通滤波器（sinc 函数）的核
M_gibbs = 51
n_gibbs = np.arange(M_gibbs)
fc_norm = 0.2  # 归一化截止频率

# 理想低通的脉冲响应（无限长 sinc）
ideal_kernel = 2 * fc_norm * np.sinc(2 * fc_norm * (n_gibbs - (M_gibbs - 1) / 2))

# 加矩形窗（= 直接截断，= firwin with boxcar）
kernel_rect = ideal_kernel * signal.windows.boxcar(M_gibbs)

# 加 Hamming 窗
kernel_hamm = ideal_kernel * signal.windows.hamming(M_gibbs)

# 加 Kaiser 窗
kernel_kaiser = ideal_kernel * signal.windows.kaiser(M_gibbs, beta=6)

# 对比三种方式
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# 时域：三种核
for ax, (k, name, color) in zip(
    [axes[0, 0], axes[0, 1], axes[1, 0]],
    [(kernel_rect, '直接截断 (矩形窗)', 'crimson'),
     (kernel_hamm, 'Hamming 窗', 'steelblue'),
     (kernel_kaiser, 'Kaiser 窗 (β=6)', 'darkorange')]):
    k_plot, name_plot, color_plot = k if isinstance(k, tuple) else (k, name, color)
    ax.stem(n_gibbs, k_plot, linefmt=f'{color_plot}-', markerfmt=f'{color_plot}o',
            basefmt='gray')
    ax.set_title(name_plot, fontsize=9)
    ax.set_ylabel('核系数')
    ax.grid(True, alpha=0.3)

# 频域：三种滤波器的频率响应
for k, name, color, ls in [
    (kernel_rect, '直接截断', 'crimson', '-'),
    (kernel_hamm, 'Hamming 窗', 'steelblue', '-'),
    (kernel_kaiser, 'Kaiser 窗 (β=6)', 'darkorange', '-')]:
    w, h = freqz(k, worN=512)
    h_db = 20 * np.log10(np.maximum(np.abs(h), 1e-15))
    axes[1, 1].plot(w / np.pi, h_db, linewidth=1.2, color=color,
                    linestyle=ls, alpha=0.8, label=name)

# 理想响应的参考线
axes[1, 1].axvline(x=fc_norm, color='black', linewidth=1, linestyle='--',
                   alpha=0.7, label=f'截止频率 = {fc_norm}π')
axes[1, 1].axhline(y=0, color='gray', linewidth=0.5)
axes[1, 1].axhline(y=-40, color='gray', linewidth=0.5, linestyle=':')
axes[1, 1].set_xlabel('归一化频率 (× π rad/sample)')
axes[1, 1].set_ylabel('幅度 (dB)')
axes[1, 1].set_title('频率响应对比 — 直接截断产生 Gibbs 振荡！')
axes[1, 1].set_ylim(-80, 5)
axes[1, 1].legend(fontsize=7)
axes[1, 1].grid(True, alpha=0.3)

fig.suptitle('Gibbs 现象 — 为什么滤波器设计必须用窗函数',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch03_fig3_gibbs_phenomenon.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch03_fig3_gibbs_phenomenon.png")

# ============================================================
# Part 4: firwin 四种滤波器设计 + freqz 分析
# ============================================================
print("\n" + "=" * 60)
print("Part 4: firwin 设计四种滤波器 + freqz 分析")
print("=" * 60)

fs = 252          # 日度金融数据
nyq = fs / 2      # 126 次/年
numtaps = 63      # 滤波器阶数

# 设计四种滤波器
filters = {}

# 低通：保留周期 > 20天（频率 < 252/20 ≈ 12.6 次/年）
filters['低通 (周期>20天)'] = firwin(
    numtaps, cutoff=12.6, window='hamming', pass_zero=True, fs=fs
)

# 高通：保留周期 < 10天（频率 > 252/10 ≈ 25.2 次/年）
filters['高通 (周期<10天)'] = firwin(
    numtaps, cutoff=25.2, window='hamming', pass_zero=False, fs=fs
)

# 带通：保留 20-60 天的周期
filters['带通 (周期20-60天)'] = firwin(
    numtaps, cutoff=[4.2, 12.6], window='hamming', pass_zero=False, fs=fs
)

# 带阻（陷波）：去掉周度效应（约5个交易日 = 252/5 ≈ 50.4 次/年）
filters['带阻 (去周度效应)'] = firwin(
    numtaps, cutoff=[40, 60], window='hamming', pass_zero=True, fs=fs
)

print(f"  采样率 fs = {fs} 次/年, Nyquist = {nyq} 次/年")
print(f"  滤波器阶数 numtaps = {numtaps}\n")

# 可视化频率响应
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, (name, b) in zip(axes.flat, filters.items()):
    w, h = freqz(b, worN=1024, fs=fs)
    h_db = 20 * np.log10(np.maximum(np.abs(h), 1e-15))

    ax.plot(w, h_db, linewidth=1.2, color='steelblue')
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axhline(y=-3, color='red', linewidth=0.5, linestyle='--',
               alpha=0.6, label='-3 dB (截止频率参考)')
    ax.axhline(y=-40, color='gray', linewidth=0.5, linestyle=':',
               alpha=0.6, label='-40 dB')

    # 标注 pass/stop 区域
    if '低通' in name:
        ax.axvspan(0, 12.6, alpha=0.08, color='green')
        ax.axvspan(25, nyq, alpha=0.08, color='red')
        ax.text(6, -10, '通带', fontsize=8, color='green', ha='center')
        ax.text(80, -10, '阻带', fontsize=8, color='red', ha='center')
    elif '高通' in name:
        ax.axvspan(0, 15, alpha=0.08, color='red')
        ax.axvspan(25.2, nyq, alpha=0.08, color='green')
    elif '带通' in name:
        ax.axvspan(4.2, 12.6, alpha=0.08, color='green')
        ax.axvspan(0, 2, alpha=0.08, color='red')
        ax.axvspan(20, nyq, alpha=0.08, color='red')
    elif '带阻' in name:
        ax.axvspan(40, 60, alpha=0.08, color='red')
        ax.axvspan(0, 30, alpha=0.08, color='green')
        ax.axvspan(70, nyq, alpha=0.08, color='green')

    ax.set_xlabel('频率 (次/年)')
    ax.set_ylabel('幅度 (dB)')
    ax.set_title(f'{name} — 频率响应', fontsize=10)
    ax.set_ylim(-80, 5)
    ax.legend(fontsize=7, loc='lower left')
    ax.grid(True, alpha=0.3)

fig.suptitle(f'firwin 四种滤波器频率响应 (numtaps={numtaps}, fs={fs}次/年)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch03_fig4_four_filters.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch03_fig4_four_filters.png")

# ---- 群延迟分析 ----
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

for ax, (name, b) in zip(axes.flat, list(filters.items())[:4]):
    freqs, gd = group_delay((b, 1), fs=fs)
    ax.plot(freqs, gd, linewidth=1.2, color='steelblue')
    ax.axhline(y=(numtaps-1)/2, color='red', linewidth=0.8, linestyle='--',
               alpha=0.7, label=f'理论群延迟 = {(numtaps-1)/2:.1f} 样本')
    ax.set_xlabel('频率 (次/年)')
    ax.set_ylabel('群延迟 (样本)')
    ax.set_title(f'{name}', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

fig.suptitle('群延迟分析 — FIR 对称核 = 常数群延迟 = 无相位失真',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch03_fig5_group_delay.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch03_fig5_group_delay.png")

# ============================================================
# Part 5: 金融实战 — 提取股价的中期趋势
# ============================================================
print("\n" + "=" * 60)
print("Part 5: 金融实战 — 低通滤波提取中期趋势")
print("=" * 60)

# 构造含多周期成分的"类股价"
np.random.seed(42)
N_days = 504  # 约 2 年的交易日
t_days = np.arange(N_days) / fs  # 单位：年

# 长期趋势 (周期 >> 100天)
trend = 50 + 15 * t_days

# 中期周期 (20-60天)
mid_cycle = 8 * np.sin(2 * np.pi * t_days * 252 / 40)

# 短期波动 (< 20天，我们想去掉的)
short_cycle = 3 * np.sin(2 * np.pi * t_days * 252 / 8)
noise = 1.5 * np.random.randn(N_days)

price = trend + mid_cycle + short_cycle + noise

# 设计低通滤波器：保留 20 天以上的趋势
cutoff_low = fs / 20  # ≈ 12.6 次/年
b_low = firwin(numtaps=63, cutoff=cutoff_low, window='hamming', fs=fs)
price_trend = filtfilt(b_low, [1], price)  # 零相位滤波

# 设计带通滤波器：只保留 20-60 天的中期周期
b_band = firwin(numtaps=63, cutoff=[fs/60, fs/20],
                window='hamming', pass_zero=False, fs=fs)
price_midcycle = filtfilt(b_band, [1], price)

# 高通：只看短期波动（< 10天）
b_high = firwin(numtaps=63, cutoff=fs/10, window='hamming',
                pass_zero=False, fs=fs)
price_short = filtfilt(b_high, [1], price)

print(f"""
  用 FIR 滤波器将股价分解为三个频率带：
    1. 低通 (周期 > 20天)   → 提取长期趋势
    2. 带通 (周期 20-60天)  → 提取中期周期
    3. 高通 (周期 < 10天)   → 短期波动 + 噪声

  对比第2章的简单均线：FIR 滤波器给你精确的频率控制！
""")

# 可视化
fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)

axes[0].plot(t_days * 252, price, linewidth=0.4, color='gray', alpha=0.7)
axes[0].plot(t_days * 252, trend + mid_cycle, linewidth=0.7, color='black',
             alpha=0.5, linestyle='--', label='真实趋势+中期周期')
axes[0].set_ylabel('价格')
axes[0].set_title('原始股价（灰） vs 真实趋势+中期周期（黑虚线）')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_days * 252, price, linewidth=0.3, color='gray', alpha=0.4)
axes[1].plot(t_days * 252, price_trend, linewidth=1.2, color='steelblue',
             label='FIR 低通滤波 (周期>20天)')
axes[1].plot(t_days * 252, trend, linewidth=0.7, color='green', linestyle='--',
             alpha=0.7, label='真实趋势')
axes[1].set_ylabel('价格')
axes[1].set_title('低通滤波器输出 — 提取长期趋势')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

axes[2].plot(t_days * 252, price_midcycle, linewidth=1.0, color='darkorange',
             label='FIR 带通滤波 (周期20-60天)')
axes[2].plot(t_days * 252, mid_cycle, linewidth=0.5, color='black',
             linestyle='--', alpha=0.6, label='真实中期周期')
axes[2].axhline(y=0, color='gray', linewidth=0.5)
axes[2].set_ylabel('偏离')
axes[2].set_title('带通滤波器输出 — 提取中期周期成分')
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

axes[3].plot(t_days * 252, price_short, linewidth=0.6, color='crimson',
             label='FIR 高通滤波 (周期<10天)')
axes[3].axhline(y=0, color='gray', linewidth=0.5)
axes[3].set_xlabel('交易日')
axes[3].set_ylabel('偏离')
axes[3].set_title('高通滤波器输出 — 短期波动 + 噪声')
axes[3].legend(fontsize=8)
axes[3].grid(True, alpha=0.3)

fig.suptitle('金融实战：用 FIR 滤波器将股价分解为不同频率成分',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch03_fig6_stock_filtering.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch03_fig6_stock_filtering.png")

# ============================================================
# Part 6: 金融实战 — numtaps 参数的影响
# ============================================================
print("\n" + "=" * 60)
print("Part 6: numtaps 参数 — 阶数越大 = 过渡带越窄 = 延迟越大")
print("=" * 60)

taps_list = [21, 41, 81, 161]
fig, axes = plt.subplots(len(taps_list) + 1, 1, figsize=(16, 10), sharex=True)

# 第1行：原始 + 真实趋势
axes[0].plot(t_days * 252, price, linewidth=0.3, color='gray', alpha=0.5)
axes[0].plot(t_days * 252, trend + mid_cycle, linewidth=0.7, color='black',
             linestyle='--', alpha=0.7, label='目标（趋势+中期周期）')
axes[0].set_ylabel('价格')
axes[0].set_title('原始股价 & 目标平滑线')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

for ax, ntaps in zip(axes[1:], taps_list):
    b = firwin(numtaps=ntaps, cutoff=cutoff_low, window='hamming', fs=fs)
    y = filtfilt(b, [1], price)
    delay = (ntaps - 1) / 2  # 群延迟（样本数）

    ax.plot(t_days * 252, price, linewidth=0.2, color='gray', alpha=0.3)
    ax.plot(t_days * 252, y, linewidth=1.0, color='steelblue',
            label=f'numtaps={ntaps} (延迟≈{delay:.0f}天)')
    ax.plot(t_days * 252, trend + mid_cycle, linewidth=0.5, color='black',
            linestyle='--', alpha=0.6)
    ax.set_ylabel('价格')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('交易日')
fig.suptitle('numtaps 参数对比 — 阶数越大，越接近目标，但延迟也越大',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch03_fig7_numtaps_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch03_fig7_numtaps_comparison.png")

# ============================================================
# Part 7: 电力实战 — 带通提取日周期 + 带阻去除谐波
# ============================================================
print("\n" + "=" * 60)
print("Part 7: 电力实战 — 带通提取日周期 & 带阻去除工频")
print("=" * 60)

# 模拟一周的电力负荷数据（15分钟一个点）
np.random.seed(88)
points_per_hour = 4        # 每15分钟一个点
fs_load = points_per_hour  # 4 次/小时
hours_total = 24 * 14      # 14 天
N_load = hours_total * points_per_hour
t_load = np.arange(N_load) / fs_load  # 单位：小时

# 构造负荷信号
# 24h 日周期
daily = 0.5 + 0.3 * np.sin(2 * np.pi * t_load / 24 - np.pi/2)
# 12h 半日周期（上午和晚上两个峰）
half_day = 0.1 * np.sin(2 * np.pi * t_load / 12 - np.pi/3)
# 长期趋势（缓慢变化）
long_trend = 0.05 * np.sin(2 * np.pi * t_load / (24*14))  # 14天周期
# 随机噪声
noise_load = 0.04 * np.random.randn(N_load)
# 添加一些"尖锐脉冲"（模拟随机用电事件）
spike_mask = np.random.rand(N_load) < 0.003
spikes = spike_mask * np.random.exponential(0.3, N_load)

load_signal = daily + half_day + long_trend + noise_load + spikes

print(f"""
  模拟 14 天电力负荷（每15分钟一个点，共 {N_load} 个点）：
    - 24h 日周期（主成分）
    - 12h 半日周期（双峰结构）
    - 14 天长周期趋势
    - 随机噪声 + 偶发脉冲

  现在用 FIR 滤波器：
    1. 带通滤波器提取 24h 周期（0.9~1.1 次/天）
    2. 低通滤波器保留趋势 + 日周期
""")

# 带通：提取 24h 周期（频率 ≈ 1/24 = 0.0417 次/小时 = 1 次/天）
# 在 15 分钟采样下：目标频率 = 1/(24*4) = 1/96 次/采样 ≈ 0.0104
f_target = 1 / 24  # 1 次/天对应的频率（次/小时）
f_band_low = 0.8 / 24   # 周期 30h（允许偏差）
f_band_high = 1.2 / 24  # 周期 20h

b_daily = firwin(numtaps=201,
                 cutoff=[f_band_low, f_band_high],
                 window='kaiser', pass_zero=False, fs=fs_load)
daily_extracted = filtfilt(b_daily, [1], load_signal)

# 低通：保留趋势（周期 > 36h）
b_smooth = firwin(numtaps=101,
                  cutoff=1/36,  # 周期 36h
                  window='hamming', fs=fs_load)
load_smooth = filtfilt(b_smooth, [1], load_signal)

# --- 带阻演示：去掉"工频干扰" ----
# 假设电网频率数据中有 50Hz 干扰
# 模拟一段含 50Hz 干扰的电网频率数据
fs_grid = 1000    # 1000 Hz 采样
t_grid = np.arange(2 * fs_grid) / fs_grid
grid_signal = 50.0 + 0.05 * np.sin(2 * np.pi * 0.5 * t_grid)  # 电网频率在 50Hz 附近
grid_signal += 0.1 * np.sin(2 * np.pi * 50 * t_grid)           # 50Hz 干扰！
grid_signal += 0.02 * np.random.randn(len(t_grid))

# 设计陷波器（带阻）去除 50Hz
b_notch = firwin(numtaps=201,
                 cutoff=[48, 52],   # 48-52 Hz 阻带
                 window='hamming', pass_zero=True, fs=fs_grid)
grid_cleaned = filtfilt(b_notch, [1], grid_signal)

# 可视化
fig, axes = plt.subplots(3, 2, figsize=(16, 12))

# 电力负荷：原始 vs 提取的日周期
plot_range = 4 * 24  # 显示前 4 天
t_plot = t_load[:plot_range*points_per_hour]
axes[0, 0].plot(t_plot, load_signal[:len(t_plot)], linewidth=0.3, color='gray',
                alpha=0.6, label='原始负荷')
axes[0, 0].plot(t_plot, daily_extracted[:len(t_plot)], linewidth=1.2,
                color='steelblue', label='带通提取的日周期')
axes[0, 0].set_xlabel('时间 (小时)')
axes[0, 0].set_ylabel('负荷')
axes[0, 0].set_title('电力负荷 — 带通滤波器提取 24h 日周期')
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True, alpha=0.3)

# 电力负荷：平滑
axes[0, 1].plot(t_plot, load_signal[:len(t_plot)], linewidth=0.3, color='gray',
                alpha=0.6, label='原始负荷')
axes[0, 1].plot(t_plot, load_smooth[:len(t_plot)], linewidth=1.2,
                color='darkorange', label='低通滤波 (周期>36h)')
axes[0, 1].set_xlabel('时间 (小时)')
axes[0, 1].set_ylabel('负荷')
axes[0, 1].set_title('电力负荷 — 低通滤波平滑（保留趋势 + 日周期）')
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(True, alpha=0.3)

# 带阻滤波器频率响应
w_notch, h_notch = freqz(b_notch, worN=1024, fs=fs_grid)
h_notch_db = 20 * np.log10(np.maximum(np.abs(h_notch), 1e-15))
axes[1, 0].plot(w_notch, h_notch_db, linewidth=1.2, color='steelblue')
axes[1, 0].axvspan(48, 52, alpha=0.1, color='red')
axes[1, 0].axhline(y=-3, color='red', linewidth=0.5, linestyle='--', alpha=0.6)
axes[1, 0].axvline(x=50, color='red', linewidth=1, linestyle='--', alpha=0.8)
axes[1, 0].set_xlabel('频率 (Hz)')
axes[1, 0].set_ylabel('幅度 (dB)')
axes[1, 0].set_title('50Hz 陷波器频率响应 — 48-52Hz 被大幅衰减')
axes[1, 0].set_ylim(-80, 5)
axes[1, 0].set_xlim(0, 100)
axes[1, 0].grid(True, alpha=0.3)

# 电网频率：滤波前后对比
plot_grid = 200  # 显示前 200 个点
axes[1, 1].plot(t_grid[:plot_grid], grid_signal[:plot_grid], linewidth=0.4,
                color='gray', alpha=0.6, label='含 50Hz 干扰')
axes[1, 1].plot(t_grid[:plot_grid], grid_cleaned[:plot_grid], linewidth=1.0,
                color='darkorange', label='陷波后')
axes[1, 1].set_xlabel('时间 (秒)')
axes[1, 1].set_ylabel('频率 (Hz)')
axes[1, 1].set_title('电网频率 — 50Hz 工频干扰去除')
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

# 滤波前后频谱对比
for ax, (sig, label, color) in zip(
    [axes[2, 0], axes[2, 1]],
    [(grid_signal, '滤波前', 'gray'),
     (grid_cleaned, '滤波后', 'darkorange')]):
    X = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(sig), 1/fs_grid)
    mask = freqs <= 100
    ax.plot(freqs[mask], X[mask], linewidth=0.8, color=color)
    ax.axvline(x=50, color='red', linewidth=1, linestyle='--', alpha=0.8)
    ax.set_xlabel('频率 (Hz)')
    ax.set_ylabel('幅度')
    ax.set_title(f'频谱 — {label}')
    ax.grid(True, alpha=0.3)

fig.suptitle('电力场景实战 — 日周期提取 & 工频干扰去除',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch03_fig8_power_filtering.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch03_fig8_power_filtering.png")

# ============================================================
# Part 8: FIRFilterDesigner 工具类
# ============================================================
print("\n" + "=" * 60)
print("Part 8: FIRFilterDesigner — 可复用的滤波器设计工具类")
print("=" * 60)


class FIRFilterDesigner:
    """
    FIR 滤波器设计与分析工具类。

    使用：
        ffd = FIRFilterDesigner(fs=252)  # 日度金融数据
        b = ffd.lowpass(cutoff_period=20, numtaps=51)
        y = ffd.apply(b, signal)

        ffd.plot_response(b, title='My Filter')
    """

    def __init__(self, fs=1.0):
        self.fs = fs
        self.nyq = fs / 2

    def _period_to_freq(self, period):
        """将周期（如"20天"）转换为频率"""
        return self.fs / period

    def lowpass(self, cutoff_period, numtaps=51, window='hamming'):
        """
        低通滤波器。

        Parameters
        ----------
        cutoff_period : float
            截止周期 —— 周期 > cutoff_period 的成分被保留
            （如 cutoff_period=20 → 保留 20 天以上的趋势）
        """
        cutoff = self._period_to_freq(cutoff_period)
        if cutoff >= self.nyq:
            raise ValueError(f"截止频率 {cutoff:.2f} >= Nyquist {self.nyq:.2f}")
        return firwin(numtaps, cutoff, window=window, pass_zero=True, fs=self.fs)

    def highpass(self, cutoff_period, numtaps=51, window='hamming'):
        """
        高通滤波器。

        Parameters
        ----------
        cutoff_period : float
            截止周期 —— 周期 < cutoff_period 的成分被保留
        """
        cutoff = self._period_to_freq(cutoff_period)
        if cutoff >= self.nyq:
            raise ValueError(f"截止频率 {cutoff:.2f} >= Nyquist {self.nyq:.2f}")
        return firwin(numtaps, cutoff, window=window, pass_zero=False, fs=self.fs)

    def bandpass(self, period_low, period_high, numtaps=51, window='hamming'):
        """
        带通滤波器。

        Parameters
        ----------
        period_low : float
            通带下限周期（周期 > period_low 被阻挡）
        period_high : float
            通带上限周期（周期 < period_high 被阻挡）
            通带 = (period_low, period_high) 之间的周期


        注：period_low < period_high（低频 → 长周期）
           对应的频率：f_low < f_high
        """
        f_low = self._period_to_freq(period_high)   # 较短的周期 → 较高频率
        f_high = self._period_to_freq(period_low)   # 较长的周期 → 较低频率
        if f_high >= self.nyq:
            raise ValueError(f"截止频率 {f_high:.2f} >= Nyquist {self.nyq:.2f}")
        return firwin(numtaps, [f_low, f_high], window=window,
                      pass_zero=False, fs=self.fs)

    def bandstop(self, period_low, period_high, numtaps=51, window='hamming'):
        """
        带阻滤波器（陷波器）。

        Parameters
        ----------
        period_low, period_high : float
            要被阻挡的周期范围
        """
        f_low = self._period_to_freq(period_high)
        f_high = self._period_to_freq(period_low)
        if f_high >= self.nyq:
            raise ValueError(f"截止频率 {f_high:.2f} >= Nyquist {self.nyq:.2f}")
        return firwin(numtaps, [f_low, f_high], window=window,
                      pass_zero=True, fs=self.fs)

    @staticmethod
    def apply(b, x, zero_phase=True):
        """
        应用滤波器。

        Parameters
        ----------
        zero_phase : bool
            True → 用 filtfilt（零相位，无延迟）
            False → 用 lfilter（因果，有延迟）
        """
        if zero_phase:
            return filtfilt(b, [1], x)
        else:
            return lfilter(b, [1], x)

    def plot_response(self, b, title='FIR Filter', ax=None):
        """绘制滤波器的频率响应和群延迟"""
        if ax is None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6))
        else:
            fig = ax.figure
            ax1, ax2 = ax

        w, h = freqz(b, worN=1024, fs=self.fs)
        h_db = 20 * np.log10(np.maximum(np.abs(h), 1e-15))

        ax1.plot(w, h_db, linewidth=1.2, color='steelblue')
        ax1.axhline(y=0, color='gray', linewidth=0.5)
        ax1.axhline(y=-3, color='red', linewidth=0.5, linestyle='--', alpha=0.6)
        ax1.set_xlabel(f'频率 (次/{self._guess_unit()})')
        ax1.set_ylabel('幅度 (dB)')
        ax1.set_title(f'{title} — 频率响应 (numtaps={len(b)})')
        ax1.set_ylim(-80, 5)
        ax1.grid(True, alpha=0.3)

        freqs, gd = group_delay((b, 1), fs=self.fs)
        ax2.plot(freqs, gd, linewidth=1.2, color='darkorange')
        ax2.axhline(y=(len(b)-1)//2, color='red', linewidth=0.8, linestyle='--',
                    alpha=0.7, label=f'群延迟 ≈ {(len(b)-1)//2} 样本')
        ax2.set_xlabel(f'频率 (次/{self._guess_unit()})')
        ax2.set_ylabel('群延迟 (样本)')
        ax2.set_title('群延迟')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        return fig

    def _guess_unit(self):
        """根据采样率猜测时间单位"""
        if self.fs >= 100:
            return '秒'
        elif self.fs >= 1:
            return '天' if self.fs <= 366 else '小时'
        else:
            return '时间单位'


# 演示 FIRFilterDesigner
ffd = FIRFilterDesigner(fs=252)  # 日度金融数据

print("\n  设计滤波器：")
print(f"    - 低通 (周期>20天)  → 提取长期趋势")
print(f"    - 带通 (周期20-60天) → 提取中期周期")
print(f"    - 高通 (周期<10天)  → 短期波动")

b_low_demo = ffd.lowpass(cutoff_period=20, numtaps=51)
b_band_demo = ffd.bandpass(period_low=20, period_high=60, numtaps=51)
b_high_demo = ffd.highpass(cutoff_period=10, numtaps=51)

# 对 Part 5 的股价数据应用
price_low = ffd.apply(b_low_demo, price)
price_band = ffd.apply(b_band_demo, price)
price_high = ffd.apply(b_high_demo, price)

fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)

axes[0].plot(price, linewidth=0.3, color='gray', alpha=0.4)
axes[0].plot(price_low, linewidth=1, color='steelblue')
axes[0].set_ylabel('价格')
axes[0].set_title('FIRFilterDesigner — 低通 (周期>20天)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(price_band, linewidth=1, color='darkorange')
axes[1].axhline(y=0, color='gray', linewidth=0.5)
axes[1].set_ylabel('偏离')
axes[1].set_title('FIRFilterDesigner — 带通 (周期20-60天)')
axes[1].grid(True, alpha=0.3)

axes[2].plot(price_high, linewidth=0.6, color='crimson')
axes[2].axhline(y=0, color='gray', linewidth=0.5)
axes[2].set_xlabel('交易日')
axes[2].set_ylabel('偏离')
axes[2].set_title('FIRFilterDesigner — 高通 (周期<10天)')
axes[2].grid(True, alpha=0.3)

fig.suptitle('FIRFilterDesigner 工具类演示', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch03_fig9_fir_designer_demo.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch03_fig9_fir_designer_demo.png")
print("  FIRFilterDesigner 类已封装完毕——用周期描述截止频率，更符合金融分析习惯。")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("第3章运行完毕！")
print("=" * 60)
print(f"""
📋 本章产出：

  笔记：doc/03_FIR滤波器设计_窗口法.md
  代码：code/ch03_FIR滤波器设计_窗口法.py
  图片：
    ch03_fig1_window_comparison.png      — 7种窗函数时域+频域对比
    ch03_fig2_kaiser_beta_sweep.png      — Kaiser β 参数扫参
    ch03_fig3_gibbs_phenomenon.png       — Gibbs 现象（为什么必须用窗）
    ch03_fig4_four_filters.png           — 四种滤波器频率响应
    ch03_fig5_group_delay.png            — 群延迟分析（常数 = 无相位失真）
    ch03_fig6_stock_filtering.png        — 股价三频带分解（低通/带通/高通）
    ch03_fig7_numtaps_comparison.png     — numtaps 参数影响
    ch03_fig8_power_filtering.png        — 电力日周期提取 + 50Hz 陷波
    ch03_fig9_fir_designer_demo.png      — FIRFilterDesigner 工具类

🎯 核心收获：
  1. 均线 = 最粗糙的低通滤波器 → FIR 设计给你精确的频率控制
  2. 窗函数 = 消除 Gibbs 振荡 → Kaiser 窗一个参数控制一切
  3. firwin(cutoff, window, fs) = 三参数搞定滤波器设计
  4. 截止频率必须 < Nyquist (fs/2) → 用 fs 参数避免归一化错误
  5. numtaps 越大 → 过渡带越窄 → 但延迟越大
  6. FIR 对称核 → 线性相位 → 群延迟常数 → 波形不失真

📖 下一站：第4章 — FIR 滤波器设计（下）Remez 最优法
  → 同等阶数下性能更好，工业级FIR设计的标配
""")
