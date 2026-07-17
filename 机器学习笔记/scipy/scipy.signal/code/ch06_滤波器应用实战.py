#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第6章 · 滤波器应用实战 — 配套代码
=========================================
本章按场景组织：每个场景一个问题、一个解决办法、一张对比图。

场景1: 离线趋势提取 — lfilter vs filtfilt 的延迟问题
场景2: 实时数据流处理 — sosfilt + zi 状态传递
场景3: 高精度时间定位 — filtfilt 的边界效应和幅度翻倍
场景4: 降采样混叠 — 为什么降采样前必须做抗混叠滤波
场景5: 高阶 IIR 数值精度 — SOS 救你一命

运行方式：
  python code/ch06_滤波器应用实战.py

依赖：
  pip install numpy scipy matplotlib
"""

import numpy as np
from scipy import signal
from scipy.signal import (butter, firwin, lfilter, filtfilt,
                          sosfilt, sosfiltfilt, sosfilt_zi,
                          tf2sos, decimate, upfirdn,
                          freqz, sosfreqz)
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.dpi': 120, 'font.size': 9,
    'axes.titlesize': 11, 'axes.labelsize': 9,
})
print("=" * 60)
print("第6章 · 滤波器应用实战 — 按场景学习")
print("=" * 60)

# ============================================================
# 场景1: 离线趋势提取 — lfilter vs filtfilt
# ============================================================
print("\n" + "=" * 60)
print("场景1: 离线趋势提取 — lfilter 有延迟, filtfilt 对齐")
print("=" * 60)

# 构造数据：一年的"类股价"日线（252个交易日）
np.random.seed(42)
fs = 252
N = 252
days = np.arange(N)

# 构造一个"真实趋势 + 噪声"的场景
true_trend = 100 + 0.05 * days + 15 * np.sin(2 * np.pi * days / 63)
price = true_trend + 8 * np.random.randn(N)

# 设计低通 FIR 滤波器：保留 >20 天的趋势
b_fir = firwin(51, cutoff=fs/20, window='hamming', fs=fs)
delay_days = (len(b_fir) - 1) / 2  # = 25 天

# 两种方式施加
trend_lfilter = lfilter(b_fir, [1], price)       # 因果，有延迟
trend_filtfilt = filtfilt(b_fir, [1], price)      # 零相位，对齐

# 计算延迟前后的 MSE
valid = slice(60, -60)
mse_lfilter = np.mean((true_trend[valid] - trend_lfilter[valid]) ** 2)
mse_filtfilt = np.mean((true_trend[valid] - trend_filtfilt[valid]) ** 2)

print(f"  FIR 滤波器长度: {len(b_fir)} → 因果延迟 = {delay_days:.0f} 天")
print(f"  lfilter MSE = {mse_lfilter:.2f}")
print(f"  filtfilt MSE = {mse_filtfilt:.2f}")

# 图：完整对比
fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)

axes[0].plot(days, price, linewidth=0.3, color='gray', alpha=0.6, label='日收盘价')
axes[0].plot(days, true_trend, linewidth=0.8, color='black', linestyle='--',
             label='真实趋势 (已知)')
axes[0].set_ylabel('价格')
axes[0].set_title('原始数据：日收盘价（灰色）+ 真实趋势（黑色虚线，模拟时已知）')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].plot(days, price, linewidth=0.2, color='gray', alpha=0.3)
axes[1].plot(days, true_trend, linewidth=0.8, color='black', linestyle='--', alpha=0.6)
axes[1].plot(days, trend_lfilter, linewidth=1.2, color='steelblue')
axes[1].axvline(x=delay_days, color='red', linewidth=1, linestyle=':')
axes[1].text(delay_days + 5, axes[1].get_ylim()[1] * 0.85,
             f'延迟 = {delay_days:.0f}天', fontsize=9, color='red')
axes[1].set_ylabel('价格')
axes[1].set_title(f'lfilter (因果) — 趋势线落后真实趋势约 {delay_days:.0f} 天（注意红虚线标注）')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

# 放大延迟区域
zoom_start = 60
zoom_end = 140
axes[2].plot(days[zoom_start:zoom_end], true_trend[zoom_start:zoom_end],
             linewidth=1.5, color='black', linestyle='--', alpha=0.7, label='真实趋势')
axes[2].plot(days[zoom_start:zoom_end], trend_lfilter[zoom_start:zoom_end],
             linewidth=1.5, color='steelblue', label='lfilter (因果)')
axes[2].plot(days[zoom_start:zoom_end], trend_filtfilt[zoom_start:zoom_end],
             linewidth=1.5, color='crimson', label='filtfilt (零相位)')
# 标注延迟
peak_idx_true = zoom_start + np.argmax(true_trend[zoom_start:zoom_end])
peak_idx_lfilter = zoom_start + np.argmax(trend_lfilter[zoom_start:zoom_end])
axes[2].axvline(x=peak_idx_true, color='black', linewidth=1, linestyle=':')
axes[2].axvline(x=peak_idx_lfilter, color='steelblue', linewidth=1, linestyle=':')
axes[2].annotate('', xy=(peak_idx_lfilter, true_trend[peak_idx_true]),
                 xytext=(peak_idx_true, true_trend[peak_idx_true]),
                 arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
axes[2].text((peak_idx_true + peak_idx_lfilter) / 2,
             true_trend[peak_idx_true] + 1,
             f'{peak_idx_lfilter - peak_idx_true}天延迟', fontsize=9,
             color='red', ha='center')
axes[2].set_xlabel('交易日')
axes[2].set_ylabel('价格')
axes[2].set_title('放大：lfilter 的峰值比真实趋势晚出现（红箭头），filtfilt 完美对齐')
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

fig.suptitle('场景1 — 离线趋势提取：lfilter 有延迟 vs filtfilt 无延迟',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch06_fig1_trend_extraction.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch06_fig1_trend_extraction.png")

# ============================================================
# 场景2: 实时数据流 — sosfilt + zi 状态传递
# ============================================================
print("\n" + "=" * 60)
print("场景2: 实时数据流 — 逐批处理 + 状态保持")
print("=" * 60)

# 模拟场景：你有一个实时传感器，每秒回传一个数据点
# 你没法等全部数据到齐——必须每收到一批就处理一批
np.random.seed(123)
fs_stream = 100
total_seconds = 10
total_points = total_seconds * fs_stream
t_stream = np.arange(total_points) / fs_stream

# 构造信号：低频有用信号 + 高频噪声
truth_signal = np.sin(2 * np.pi * 2 * t_stream)                # 2 Hz 有用信号
observed = truth_signal + 0.5 * np.random.randn(total_points)  # 叠加噪声

# 设计实时滤波器（Butterworth 3阶低通, 截止 5 Hz）
sos = butter(3, 5, btype='low', fs=fs_stream, output='sos')

# ---- 错误做法：每批重新初始化（演示瞬态问题） ----
# 模拟"不知道有 zi"的新手做法
chunk_size = 50
y_wrong = np.zeros(total_points)
for start in range(0, total_points, chunk_size):
    end = min(start + chunk_size, total_points)
    y_wrong[start:end] = sosfilt(sos, observed[start:end])
    # ↑ 每批都从零状态开始 → 每批开头都有瞬态

# ---- 正确做法：保持 zi 状态 ----
y_correct = np.zeros(total_points)
zi = sosfilt_zi(sos) * observed[0]  # 稳态初始条件
for start in range(0, total_points, chunk_size):
    end = min(start + chunk_size, total_points)
    y_correct[start:end], zi = sosfilt(sos, observed[start:end], zi=zi)
    # ↑ zi 随批次更新 → 状态无缝衔接

print(f"  总数据点: {total_points}, 每批 {chunk_size} 点, 共 {total_points//chunk_size} 批")

# 图：放大前3批数据（150个点），看清瞬态
fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)

plot_range = slice(0, 200)

axes[0].plot(t_stream[plot_range], observed[plot_range],
             linewidth=0.3, color='gray', alpha=0.5, label='观测 (含噪声)')
axes[0].plot(t_stream[plot_range], truth_signal[plot_range],
             linewidth=0.6, color='black', linestyle='--', alpha=0.6, label='真实信号')
axes[0].plot(t_stream[plot_range], y_wrong[plot_range],
             linewidth=1.0, color='crimson', alpha=0.8,
             label='错误做法：每批重置状态')
# 标记批次边界
for b in range(0, 200, chunk_size):
    axes[0].axvline(x=t_stream[b], color='red', linewidth=0.5, linestyle=':', alpha=0.6)
axes[0].set_ylabel('幅值')
axes[0].set_title('错误做法 — 每批从零状态开始，批次接头处出现跳变瞬态（红虚线=批次边界）')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_stream[plot_range], observed[plot_range],
             linewidth=0.3, color='gray', alpha=0.5, label='观测 (含噪声)')
axes[1].plot(t_stream[plot_range], truth_signal[plot_range],
             linewidth=0.6, color='black', linestyle='--', alpha=0.6, label='真实信号')
axes[1].plot(t_stream[plot_range], y_correct[plot_range],
             linewidth=1.0, color='steelblue', alpha=0.8,
             label='正确做法：zi 状态传递')
for b in range(0, 200, chunk_size):
    axes[1].axvline(x=t_stream[b], color='blue', linewidth=0.5, linestyle=':', alpha=0.4)
axes[1].set_xlabel('时间 (秒)')
axes[1].set_ylabel('幅值')
axes[1].set_title('正确做法 — 状态在批次间传递，输出连续平滑（接头处无跳变）')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

fig.suptitle('场景2 — 实时流式滤波：zi 状态传递消除瞬态',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch06_fig2_streaming.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch06_fig2_streaming.png")

# ============================================================
# 场景3: 高精度时间定位 — 边界效应和幅度翻倍
# ============================================================
print("\n" + "=" * 60)
print("场景3: 高精度时间定位 — filtfilt 的代价")
print("=" * 60)

# 模拟场景：你有一个脉冲信号，需要精确知道峰值时刻
# 比如：电网频率突然跳变的时间点、EEG 事件相关电位
np.random.seed(77)
N_impulse = 300
t_impulse = np.arange(N_impulse) / 100.0

# 在 t=1.5s 处有一个脉冲事件
impulse_signal = np.zeros(N_impulse)
impulse_signal[145:155] = 1.0  # 脉冲在位置 150 (t=1.5s)
impulse_signal += 0.05 * np.random.randn(N_impulse)

# 设计滤波器
sos_pulse = butter(3, 15, btype='low', fs=100, output='sos')

# 因果滤波（会有延迟）
y_causal = sosfilt(sos_pulse, impulse_signal)

# 零相位滤波（无延迟，但边界有伪影）
y_zero = sosfiltfilt(sos_pulse, impulse_signal)

# 找到峰值位置
peak_orig = np.argmax(impulse_signal)
peak_causal = np.argmax(y_causal)
peak_zero = np.argmax(y_zero)

print(f"  原始峰值位置: {peak_orig} (t={peak_orig/100:.2f}s)")
print(f"  因果滤波后峰值: {peak_causal} (t={peak_causal/100:.2f}s) → 延迟 {peak_causal-peak_orig} 样本")
print(f"  零相位滤波后峰值: {peak_zero} (t={peak_zero/100:.2f}s) → 延迟 {peak_zero-peak_orig} 样本")

# 图
fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)

axes[0].plot(t_impulse, impulse_signal, linewidth=0.8, color='gray')
axes[0].axvline(x=t_impulse[peak_orig], color='black', linewidth=1, linestyle='--',
                label=f'原始峰值 (t={peak_orig/100:.2f}s)')
axes[0].set_ylabel('幅值')
axes[0].set_title('原始脉冲信号 — 峰值在 t≈1.5s')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_impulse, y_causal, linewidth=1.0, color='steelblue')
axes[1].axvline(x=t_impulse[peak_causal], color='steelblue', linewidth=1, linestyle='--',
                label=f'因果滤波峰值 (t={peak_causal/100:.2f}s)')
axes[1].axvline(x=t_impulse[peak_orig], color='black', linewidth=0.8, linestyle=':', alpha=0.5)
axes[1].set_ylabel('幅值')
axes[1].set_title(f'因果滤波 — 峰值延迟了 {peak_causal-peak_orig} 个样本')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

axes[2].plot(t_impulse, y_zero, linewidth=1.0, color='crimson')
axes[2].axvline(x=t_impulse[peak_zero], color='crimson', linewidth=1, linestyle='--',
                label=f'零相位峰值 (t={peak_zero/100:.2f}s)')
axes[2].axvline(x=t_impulse[peak_orig], color='black', linewidth=0.8, linestyle=':', alpha=0.5)
# 标记边界
margin = 3 * 3 * 2
axes[2].axvspan(0, t_impulse[margin], alpha=0.1, color='red')
axes[2].axvspan(t_impulse[-margin], t_impulse[-1], alpha=0.1, color='red')
axes[2].text(t_impulse[margin]/2, axes[2].get_ylim()[1]*0.5, '边界\n不可靠', fontsize=8,
             ha='center', color='red')
axes[2].set_xlabel('时间 (秒)')
axes[2].set_ylabel('幅值')
axes[2].set_title(f'零相位滤波 — 峰值精确对齐！但注意红色区域（边界伪影）不可用')
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

fig.suptitle('场景3 — 时间精度要求高时，filtfilt 保位置，但注意边界和幅度翻倍',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch06_fig3_time_precision.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch06_fig3_time_precision.png")

# ============================================================
# 场景4: 降采样 — 为什么必须先做抗混叠滤波
# ============================================================
print("\n" + "=" * 60)
print("场景4: 降采样 — 抗混叠滤波的威力")
print("=" * 60)

# 模拟场景：你有一个 1000 Hz 采样的信号，想降到 100 Hz
# 原信号包含：30 Hz（保留）+ 80 Hz（超过新 Nyquist=50Hz，会混叠！）
np.random.seed(55)
fs_orig = 1000
t_orig = np.arange(4000) / fs_orig

# 构造信号
sig_low = np.sin(2 * np.pi * 30 * t_orig)     # 30 Hz — 希望保留
sig_high = 0.4 * np.sin(2 * np.pi * 80 * t_orig)  # 80 Hz — 超过新 Nyquist(50Hz)
sig = sig_low + sig_high

# 新采样率 100 Hz → Nyquist = 50 Hz
# 80 Hz > 50 Hz → 如果不做抗混叠，80 Hz 会伪造成 20 Hz (|100-80|=20) 的假信号！

# ---- 错误做法：直接降采样（每10个点取1个） ----
sig_down_bad = sig[::10]

# ---- 正确做法：先抗混叠滤波再降采样 ----
# 用 decimate 一步到位
sig_down_good = decimate(sig, q=10, n=40, ftype='fir')

print(f"  原始信号：30 Hz (要保留) + 80 Hz (超过新Nyquist 50Hz)")
print(f"  直接降采样 → 80Hz 混叠为 {abs(100-80)} Hz 的假信号！")
print(f"  decimate → 先滤掉 >50Hz，再降采样 → 安全")

# 图：频谱对比
fig, axes = plt.subplots(3, 1, figsize=(16, 8))

# 原始信号频谱
freqs_orig = np.fft.rfftfreq(len(sig), 1/fs_orig)
X_orig = np.abs(np.fft.rfft(sig))
mask_orig = freqs_orig <= 150
axes[0].plot(freqs_orig[mask_orig], X_orig[mask_orig], linewidth=0.8, color='gray')
axes[0].axvline(x=30, color='green', linewidth=1, linestyle='--', alpha=0.7, label='30Hz (要保留)')
axes[0].axvline(x=80, color='red', linewidth=1, linestyle='--', alpha=0.7, label='80Hz (要滤掉)')
axes[0].axvline(x=50, color='orange', linewidth=0.8, linestyle=':', alpha=0.7, label='新Nyquist=50Hz')
axes[0].set_ylabel('幅度')
axes[0].set_title(f'原始信号频谱 (fs={fs_orig}Hz) — 30Hz 和 80Hz 共存')
axes[0].legend(fontsize=7)
axes[0].grid(True, alpha=0.3)

# 错误降采样的频谱
fs_new = 100
freqs_bad = np.fft.rfftfreq(len(sig_down_bad), 1/fs_new)
X_bad = np.abs(np.fft.rfft(sig_down_bad))
mask_bad = freqs_bad <= 50
axes[1].plot(freqs_bad[mask_bad], X_bad[mask_bad], linewidth=0.8, color='crimson')
axes[1].axvline(x=30, color='green', linewidth=1, linestyle='--', alpha=0.7, label='30Hz (正确的)')
axes[1].axvline(x=20, color='red', linewidth=1, linestyle='--', alpha=0.7,
                label='20Hz (混叠的假信号!!)')
axes[1].set_ylabel('幅度')
axes[1].set_title('直接降采样（无抗混叠）→ 80Hz 混叠成 20Hz 假峰！')
axes[1].legend(fontsize=7)
axes[1].grid(True, alpha=0.3)

# 正确降采样的频谱
freqs_good = np.fft.rfftfreq(len(sig_down_good), 1/fs_new)
X_good = np.abs(np.fft.rfft(sig_down_good))
mask_good = freqs_good <= 50
axes[2].plot(freqs_good[mask_good], X_good[mask_good], linewidth=0.8, color='steelblue')
axes[2].axvline(x=30, color='green', linewidth=1, linestyle='--', alpha=0.7, label='30Hz (正确保留)')
axes[2].set_xlabel('频率 (Hz)')
axes[2].set_ylabel('幅度')
axes[2].set_title('decimate（先抗混叠再降采样）→ 80Hz 先被滤掉，只剩 30Hz ✓')
axes[2].legend(fontsize=7)
axes[2].grid(True, alpha=0.3)

fig.suptitle('场景4 — 降采样前不滤波 = 引入假信号（混叠）',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch06_fig4_antialias.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch06_fig4_antialias.png")

# ============================================================
# 场景5: 高阶 IIR 数值精度 — SOS 救场
# ============================================================
print("\n" + "=" * 60)
print("场景5: 高阶 IIR — 为什么 (b,a) 格式会崩溃")
print("=" * 60)

# 用不同阶数的 Butterworth 对比 (b,a) vs SOS
np.random.seed(10)
test_data = np.random.randn(1000)

orders = [4, 6, 8, 10, 14]
max_errors = []

print(f"\n  {'阶数':<6s} {'(b,a) vs SOS 最大差异':<25s} {'安全性'}")
print(f"  {'-'*45}")

for N in orders:
    # (b, a) 格式
    b_ba, a_ba = butter(N, 10, btype='low', fs=200, output='ba')
    y_ba = lfilter(b_ba, a_ba, test_data)

    # SOS 格式
    sos = butter(N, 10, btype='low', fs=200, output='sos')
    y_sos = sosfilt(sos, test_data)

    diff = np.max(np.abs(y_ba - y_sos))
    max_errors.append(diff)

    status = '✓ 安全' if diff < 1e-8 else '✗ 不可靠！' if diff > 1e-4 else '⚠ 有风险'
    print(f"  {N:<6d} {diff:<25.2e} {status}")

# 图：错误 vs 阶数
fig, ax = plt.subplots(figsize=(14, 4))
ax.semilogy(orders, max_errors, 'o-', linewidth=2, markersize=10, color='steelblue')
ax.axhline(y=1e-10, color='gray', linewidth=0.5, linestyle=':', alpha=0.5,
           label='浮点精度极限 (~1e-15)')
ax.axhline(y=1e-6, color='orange', linewidth=0.8, linestyle='--', alpha=0.7,
           label='1e-6 — 开始可见')
ax.axhline(y=1e-4, color='red', linewidth=0.8, linestyle='--', alpha=0.7,
           label='1e-4 — 明显不可靠')
ax.set_xlabel('滤波器阶数 N')
ax.set_ylabel('(b,a) 与 SOS 的最大差异')
ax.set_title('场景5 — 高阶 IIR：(b,a) 误差随阶数指数增长，SOS 始终精确')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 标注各阶数的安全级别
for N, err in zip(orders, max_errors):
    color = 'green' if err < 1e-8 else 'orange' if err < 1e-4 else 'red'
    ax.annotate(f'N={N}', (N, err), textcoords="offset points",
                xytext=(0, 12), ha='center', fontsize=8, color=color)

plt.tight_layout()
plt.savefig('code/ch06_fig5_sos_stability.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch06_fig5_sos_stability.png")

# 图：直接看波形 — N=14 时 (b,a) 的灾难
N_bad = 14
b_ba, a_ba = butter(N_bad, 10, btype='low', fs=200, output='ba')
sos_bad = butter(N_bad, 10, btype='low', fs=200, output='sos')
y_ba_bad = lfilter(b_ba, a_ba, test_data)
y_sos_bad = sosfilt(sos_bad, test_data)

fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
plot_n = 200
axes[0].plot(test_data[:plot_n], linewidth=0.3, color='gray', alpha=0.4, label='原始')
axes[0].plot(y_sos_bad[:plot_n], linewidth=1.0, color='steelblue', label='SOS — 稳定')
axes[0].plot(y_ba_bad[:plot_n], linewidth=1.0, color='crimson', alpha=0.7,
             label='(b,a) — 数值灾难')
axes[0].set_ylabel('幅值')
axes[0].set_title(f'{N_bad}阶 Butterworth — SOS 平稳 vs (b,a) 漂移发散')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].plot(y_ba_bad[:plot_n] - y_sos_bad[:plot_n], linewidth=0.8, color='crimson')
axes[1].axhline(y=0, color='gray', linewidth=0.5)
axes[1].set_xlabel('样本序号')
axes[1].set_ylabel('差异')
axes[1].set_title('(b,a) 与 SOS 的差异 — 误差累积放大')
axes[1].grid(True, alpha=0.3)

fig.suptitle(f'场景5 — {N_bad}阶 IIR：为什么必须用 SOS',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch06_fig6_sos_waveform.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch06_fig6_sos_waveform.png")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("第6章运行完毕！")
print("=" * 60)
print(f"""
📋 本章产出：

  笔记：doc/06_滤波器应用实战.md（按场景组织）
  代码：code/ch06_滤波器应用实战.py（按场景组织）
  图片：
    ch06_fig1_trend_extraction.png    — 场景1: lfilter延迟 vs filtfilt对齐
    ch06_fig2_streaming.png           — 场景2: 实时流式 + zi状态传递
    ch06_fig3_time_precision.png      — 场景3: 高精度时间定位的边界问题
    ch06_fig4_antialias.png           — 场景4: 降采样混叠 vs 抗混叠保护
    ch06_fig5_sos_stability.png       — 场景5: (b,a)误差 vs 阶数
    ch06_fig6_sos_waveform.png        — 场景5: 14阶IIR的数值灾难

🎯 五个场景记住五个要点：
  场景1 → 离线趋势提取用 filtfilt（对齐），回测用 lfilter（因果）
  场景2 → 实时流必须传 zi（状态），否则每批开头有瞬态
  场景3 → 高精度时间定位用 filtfilt，但要丢弃边界+注意幅度翻倍
  场景4 → 降采样必须先做抗混叠滤波（decimate帮你自动做了）
  场景5 → IIR 阶数>4就开始考虑 SOS，>6 必须用 SOS

📖 下一站：第7章 — FFT 与功率谱密度估计
  → 从时域走进频域，用频谱"看见"信号中隐藏的周期成分
""")
