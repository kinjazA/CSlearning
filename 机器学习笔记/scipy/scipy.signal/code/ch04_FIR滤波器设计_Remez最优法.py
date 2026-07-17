#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第4章 · FIR 滤波器设计（下）— Remez 最优法 — 配套代码
=========================================================
场景定位：金融时间序列 & 电力负荷数据
核心主题：用 Remez/Parks-McClellan 算法设计等波纹最优 FIR 滤波器

演示内容：
  1. remez vs firwin：同阶数下过渡带宽度对比
  2. weight 参数——控制通带 vs 阻带精度比
  3. 多带滤波器设计（firwin 做不到的）
  4. 最小相位 FIR — 减少延迟
  5. 设计→分析→验证 完整管线
  6. 金融实战：remze 低通提取趋势 vs SMA vs firwin
  7. 电力实战：多带滤波器分离负荷分量
  8. 封装 OptimalFilterDesigner 工具类

运行方式：
  python code/ch04_FIR滤波器设计_Remez最优法.py

依赖：
  pip install numpy scipy matplotlib
"""

import numpy as np
from scipy import signal
from scipy.signal import (remez, firwin, firwin2, freqz, group_delay,
                          lfilter, filtfilt, minimum_phase)
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
print("第4章 · FIR 滤波器设计（下）— Remez 最优法")
print("=" * 60)

# ============================================================
# Part 1: remez vs firwin — 同阶数正面 PK
# ============================================================
print("\n" + "=" * 60)
print("Part 1: remez vs firwin — 同阶数下谁更强？")
print("=" * 60)

fs = 252       # 日度金融数据
nyq = fs / 2
numtaps = 31   # 故意用较低的阶数，放大差异

# --- 设计同一个低通滤波器 ---
# 通带：0~10 次/年（周期 > 25天）
# 阻带：15~126 次/年（周期 < 17天）
# 过渡带：10~15 次/年

# firwin（窗口法）
b_firwin = firwin(numtaps, cutoff=12.5, window='hamming', fs=fs)

# remez（等波纹法）
b_remez = remez(numtaps,
                bands=[0, 10, 15, nyq],
                desired=[1, 0],
                weight=[1, 5],
                fs=fs)

# --- 频率响应对比 ---
w, h_firwin = freqz(b_firwin, worN=2048, fs=fs)
w, h_remez = freqz(b_remez, worN=2048, fs=fs)

h_firwin_db = 20 * np.log10(np.maximum(np.abs(h_firwin), 1e-15))
h_remez_db = 20 * np.log10(np.maximum(np.abs(h_remez), 1e-15))

fig, axes = plt.subplots(2, 2, figsize=(16, 8))

# 频率响应（宽频）
axes[0, 0].plot(w, h_firwin_db, linewidth=1.2, color='steelblue',
                alpha=0.8, label='firwin (hamming)')
axes[0, 0].plot(w, h_remez_db, linewidth=1.2, color='crimson',
                alpha=0.8, label='remez (equiripple)')
axes[0, 0].axvspan(0, 10, alpha=0.06, color='green')
axes[0, 0].axvspan(15, nyq, alpha=0.06, color='red')
axes[0, 0].axvline(x=10, color='green', linewidth=0.8, linestyle='--', alpha=0.5)
axes[0, 0].axvline(x=15, color='red', linewidth=0.8, linestyle='--', alpha=0.5)
axes[0, 0].axhline(y=0, color='gray', linewidth=0.5)
axes[0, 0].axhline(y=-40, color='gray', linewidth=0.5, linestyle=':')
axes[0, 0].set_xlabel('频率 (次/年)')
axes[0, 0].set_ylabel('幅度 (dB)')
axes[0, 0].set_title(f'频率响应全览 (numtaps={numtaps})')
axes[0, 0].set_ylim(-80, 5)
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True, alpha=0.3)

# 放大过渡带区域
axes[0, 1].plot(w, h_firwin_db, linewidth=1.5, color='steelblue', alpha=0.8, label='firwin')
axes[0, 1].plot(w, h_remez_db, linewidth=1.5, color='crimson', alpha=0.8, label='remez')
axes[0, 1].axvline(x=10, color='green', linewidth=0.8, linestyle='--', alpha=0.5,
                   label='通带边界')
axes[0, 1].axvline(x=15, color='red', linewidth=0.8, linestyle='--', alpha=0.5,
                   label='阻带边界')
axes[0, 1].axhline(y=-3, color='gray', linewidth=0.5, linestyle=':')
axes[0, 1].set_xlabel('频率 (次/年)')
axes[0, 1].set_ylabel('幅度 (dB)')
axes[0, 1].set_title(f'过渡带放大 — remez 过渡带明显更窄')
axes[0, 1].set_xlim(5, 20)
axes[0, 1].set_ylim(-50, 2)
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(True, alpha=0.3)

# 通带放大
axes[1, 0].plot(w, h_firwin_db, linewidth=1.5, color='steelblue', alpha=0.8)
axes[1, 0].plot(w, h_remez_db, linewidth=1.5, color='crimson', alpha=0.8)
axes[1, 0].axhline(y=0, color='gray', linewidth=0.5)
axes[1, 0].set_xlabel('频率 (次/年)')
axes[1, 0].set_ylabel('幅度 (dB)')
axes[1, 0].set_title('通带放大 — 看纹波结构')
axes[1, 0].set_xlim(0, 10)
axes[1, 0].set_ylim(-2, 0.5)
axes[1, 0].grid(True, alpha=0.3)

# 阻带放大
axes[1, 1].plot(w, h_firwin_db, linewidth=1.5, color='steelblue', alpha=0.8)
axes[1, 1].plot(w, h_remez_db, linewidth=1.5, color='crimson', alpha=0.8)
axes[1, 1].axhline(y=-40, color='gray', linewidth=0.5, linestyle=':')
axes[1, 1].set_xlabel('频率 (次/年)')
axes[1, 1].set_ylabel('幅度 (dB)')
axes[1, 1].set_title('阻带放大 — remez 阻带均等振荡（等波纹！）')
axes[1, 1].set_xlim(15, 60)
axes[1, 1].set_ylim(-80, -20)
axes[1, 1].grid(True, alpha=0.3)

# 定量标注
# 找 -3dB 截止频率
def find_cutoff(w, h_db, target=-3):
    """找最接近 target dB 的频率"""
    idx = np.argmin(np.abs(h_db - target))
    return w[idx]

cutoff_firwin = find_cutoff(w, h_firwin_db)
cutoff_remez = find_cutoff(w, h_remez_db)
# 找 -40dB 频率（阻带入口）
mask_stop = w > 12
idx_stop_firwin = np.argmin(np.abs(h_firwin_db[mask_stop] + 40))
stop_firwin = w[mask_stop][idx_stop_firwin]
idx_stop_remez = np.argmin(np.abs(h_remez_db[mask_stop] + 40))
stop_remez = w[mask_stop][idx_stop_remez]

print(f"""
  numtaps = {numtaps}

  滤波器          -3dB 截止       过渡带宽度        相对的过渡带
  ─────────────────────────────────────────────────────────
  firwin (hamming)   {cutoff_firwin:.1f} 次/年    ≈{cutoff_firwin-10:.1f} 次/年       基准
  remez (equiripple) {cutoff_remez:.1f} 次/年    ≈{cutoff_remez-10:.1f} 次/年       remez 窄约 {((cutoff_firwin-10)-(cutoff_remez-10))/(cutoff_firwin-10)*100:.0f}%

  → remez 在同阶数下过渡带明显更窄！
""")

plt.tight_layout()
plt.savefig('code/ch04_fig1_remez_vs_firwin.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch04_fig1_remez_vs_firwin.png")

# ============================================================
# Part 2: weight 参数 — 控制精度分配
# ============================================================
print("\n" + "=" * 60)
print("Part 2: weight 参数 — 控制通带 vs 阻带精度比")
print("=" * 60)

weight_configs = [
    ([1, 1], '通带=阻带 同等重要'),
    ([1, 10], '阻带精度优先 (10倍)'),
    ([10, 1], '通带精度优先 (10倍)'),
    ([1, 50], '阻带极高精度 (50倍)'),
]

fig, axes = plt.subplots(2, 2, figsize=(16, 9))

for ax, (weight, desc) in zip(axes.flat, weight_configs):
    b = remez(numtaps=31,
              bands=[0, 10, 15, nyq],
              desired=[1, 0],
              weight=weight,
              fs=fs)
    w, h = freqz(b, worN=2048, fs=fs)
    h_db = 20 * np.log10(np.maximum(np.abs(h), 1e-15))

    ax.plot(w, h_db, linewidth=1.2, color='steelblue')
    ax.axvspan(0, 10, alpha=0.06, color='green')
    ax.axvspan(15, nyq, alpha=0.06, color='red')
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axhline(y=-40, color='gray', linewidth=0.5, linestyle=':')

    # 找通带最大纹波和阻带最小衰减
    passband_mask = w <= 10
    stopband_mask = w >= 15
    passband_ripple = np.max(h_db[passband_mask]) - np.min(h_db[passband_mask])
    stopband_atten = -np.max(h_db[stopband_mask])

    ax.set_xlabel('频率 (次/年)')
    ax.set_ylabel('幅度 (dB)')
    ax.set_title(f'weight={weight} — {desc}\n'
                 f'通带纹波={passband_ripple:.2f}dB, 阻带衰减={stopband_atten:.0f}dB',
                 fontsize=9)
    ax.set_ylim(-100, 5)
    ax.grid(True, alpha=0.3)

fig.suptitle('weight 参数扫参 — 精度是一种可以"分配"的资源',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch04_fig2_weight_sweep.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch04_fig2_weight_sweep.png")
print("""
  weight 参数指南：
    [1, 1]   → 通带和阻带同等对待
    [1, 10]  → 阻带精度优先（噪声要尽可能去掉）← 金融趋势提取推荐
    [10, 1]  → 通带精度优先（信号不能失真）
    [1, 50]  → 阻带极高精度（窄带干扰去除）  ← 工频陷波推荐
""")

# ============================================================
# Part 3: 多带滤波器 — remez 的独家本领
# ============================================================
print("\n" + "=" * 60)
print("Part 3: 多带滤波器 — firwin 做不到的事")
print("=" * 60)

# 设计一个三带滤波器
# 带1 (0-2 次/年)：长期趋势 — gain=1
# 带2 (5-12 次/年)：中期周期 — gain=0 (阻挡)
# 带3 (20-60 次/年)：短期信号 — gain=0.5 (衰减一半)
bands_multi = [0, 2, 5, 12, 20, 60]  # 注意间隙是过渡带
desired_multi = [1, 0, 0.5]
weight_multi = [1, 5, 2]

b_multi = remez(numtaps=63,
                bands=bands_multi,
                desired=desired_multi,
                weight=weight_multi,
                fs=252)

w_multi, h_multi = freqz(b_multi, worN=2048, fs=252)
h_multi_db = 20 * np.log10(np.maximum(np.abs(h_multi), 1e-15))

print(f"""
  三带滤波器 (numtaps=63):
    带1: 0-2 次/年 (周期>126天) → gain=1  ← 保留长期趋势
    带2: 5-12 次/年 (周期21-63天) → gain=0  ← 阻挡中期波动
    带3: 20-60 次/年 (周期4-13天) → gain=0.5 ← 衰减后保留短期信号

    这在 firwin 中无法一步完成——必须用 remez。
""")

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(w_multi, h_multi_db, linewidth=1.2, color='steelblue')
# 标注三个频带
ax.axvspan(0, 2, alpha=0.1, color='green')
ax.axvspan(5, 12, alpha=0.1, color='red')
ax.axvspan(20, 60, alpha=0.1, color='orange')
ax.text(1, -5, 'gain=1\n长期趋势', fontsize=8, ha='center', color='green')
ax.text(8.5, -45, 'gain=0\n(阻挡)', fontsize=8, ha='center', color='red')
ax.text(40, -10, 'gain=0.5\n短期信号', fontsize=8, ha='center', color='orange')
# 参考线
for g in [0, -6]:  # -6dB ≈ 0.5 gain
    ax.axhline(y=g, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
ax.set_xlabel('频率 (次/年)')
ax.set_ylabel('幅度 (dB)')
ax.set_title('三带滤波器 — remez 多带设计示例', fontsize=12, fontweight='bold')
ax.set_ylim(-70, 5)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('code/ch04_fig3_multiband.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch04_fig3_multiband.png")

# ============================================================
# Part 4: 最小相位 FIR — 减少延迟
# ============================================================
print("\n" + "=" * 60)
print("Part 4: 最小相位 FIR — 延迟从 25 天降到...")
print("=" * 60)

# 设计一个线性相位 FIR
b_linear = firwin(numtaps=51, cutoff=10, window='hamming', fs=252)

# 转最小相位
b_minphase = minimum_phase(b_linear, method='homomorphic')

# 对比
fig, axes = plt.subplots(2, 2, figsize=(16, 8))

# 时域：脉冲响应
n_linear = np.arange(len(b_linear))
n_minphase = np.arange(len(b_minphase))
axes[0, 0].stem(n_linear, b_linear, linefmt='steelblue-', markerfmt='steelblueo',
                basefmt='gray', label=f'线性相位 FIR (len={len(b_linear)})')
axes[0, 0].stem(n_minphase, b_minphase, linefmt='crimson-', markerfmt='crimsono',
                basefmt='gray', label=f'最小相位 FIR (len={len(b_minphase)})')
axes[0, 0].set_xlabel('样本序号')
axes[0, 0].set_ylabel('核系数')
axes[0, 0].set_title('时域脉冲响应 — 最小相位把能量集中在前面')
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True, alpha=0.3)

# 频域：幅度响应（应该几乎相同）
w, h_lin = freqz(b_linear, worN=1024, fs=252)
w, h_minp = freqz(b_minphase, worN=1024, fs=252)
axes[0, 1].plot(w, 20 * np.log10(np.maximum(np.abs(h_lin), 1e-15)),
                linewidth=1.2, color='steelblue', alpha=0.8, label='线性相位')
axes[0, 1].plot(w, 20 * np.log10(np.maximum(np.abs(h_minp), 1e-15)),
                linewidth=1.2, color='crimson', alpha=0.8, label='最小相位')
axes[0, 1].set_xlabel('频率 (次/年)')
axes[0, 1].set_ylabel('幅度 (dB)')
axes[0, 1].set_title('幅度响应 — 几乎完全相同')
axes[0, 1].set_ylim(-80, 5)
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(True, alpha=0.3)

# 群延迟对比
freqs_lin, gd_lin = group_delay((b_linear, 1), fs=252)
freqs_minp, gd_minp = group_delay((b_minphase, 1), fs=252)
axes[1, 0].plot(freqs_lin, gd_lin, linewidth=1.2, color='steelblue',
                label=f'线性相位 (均值≈{np.mean(gd_lin):.1f}样本)')
axes[1, 0].plot(freqs_minp, gd_minp, linewidth=1.2, color='crimson',
                label=f'最小相位 (均值≈{np.mean(gd_minp):.1f}样本)')
axes[1, 0].set_xlabel('频率 (次/年)')
axes[1, 0].set_ylabel('群延迟 (样本)')
axes[1, 0].set_title('群延迟对比 — 最小相位的延迟大幅降低')
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

# 相位响应
phase_lin = np.unwrap(np.angle(h_lin))
phase_minp = np.unwrap(np.angle(h_minp))
axes[1, 1].plot(w, phase_lin, linewidth=1.2, color='steelblue', label='线性相位')
axes[1, 1].plot(w, phase_minp, linewidth=1.2, color='crimson', label='最小相位')
axes[1, 1].set_xlabel('频率 (次/年)')
axes[1, 1].set_ylabel('相位 (rad)')
axes[1, 1].set_title('相位响应 — 线性相位是直线，最小相位不是')
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

fig.suptitle('最小相位 FIR — 用相位线性换延迟降低', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch04_fig4_minimum_phase.png', dpi=150, bbox_inches='tight')
plt.close()

mean_lin_delay = np.mean(gd_lin)
mean_minp_delay = np.mean(gd_minp)
print(f"""
  线性相位 FIR: 长度={len(b_linear)}, 群延迟≈{mean_lin_delay:.1f} 样本 ({mean_lin_delay/252*365:.0f} 天)
  最小相位 FIR: 长度={len(b_minphase)}, 群延迟≈{mean_minp_delay:.1f} 样本 ({mean_minp_delay/252*365:.0f} 天)
  → 延迟减少了约 {(1 - mean_minp_delay/mean_lin_delay)*100:.0f}%
""")
print("  → 已保存 ch04_fig4_minimum_phase.png")

# ============================================================
# Part 5: 金融实战 — remez 低通提取趋势（三方法对比）
# ============================================================
print("\n" + "=" * 60)
print("Part 5: 金融实战 — SMA vs firwin vs remez 趋势提取")
print("=" * 60)

# 构造含噪声的股价
np.random.seed(42)
N_days = 504
t_days = np.arange(N_days) / 252
trend = 50 + 20 * t_days + 5 * np.sin(2 * np.pi * t_days)
noise = 5 * np.random.randn(N_days)
price = trend + noise

# 三种方法提取趋势
# 1) SMA-20
sma_kernel = np.ones(20) / 20
trend_sma = np.convolve(price, sma_kernel, mode='same')

# 2) firwin 低通
b_firwin_trend = firwin(51, cutoff=252/25, window='hamming', fs=252)
trend_firwin = filtfilt(b_firwin_trend, [1], price)

# 3) remez 低通
b_remez_trend = remez(31,
                      bands=[0, 252/30, 252/15, nyq],
                      desired=[1, 0],
                      weight=[1, 10],
                      fs=252)
trend_remez = filtfilt(b_remez_trend, [1], price)

# 可视化
fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)

for ax, (y, name, color) in zip(
    axes,
    [(trend_sma, 'SMA-20 (20天简单均线)', 'darkorange'),
     (trend_firwin, 'firwin 低通 (numtaps=51)', 'steelblue'),
     (trend_remez, 'remez 低通 (numtaps=31)', 'crimson')]):
    ax.plot(price, linewidth=0.25, color='gray', alpha=0.4, label='原始股价')
    ax.plot(trend, linewidth=0.8, color='black', linestyle='--',
            alpha=0.6, label='真实趋势')
    ax.plot(y, linewidth=1.2, color=color, label=name, alpha=0.9)
    ax.set_ylabel('价格')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

axes[0].set_title('SMA-20 — 简单但有延迟和边界问题')
axes[1].set_title('firwin 低通 — 精确频率控制，零相位（filtfilt）')
axes[2].set_title('remez 低通 — 同等效果用更低阶数（31 vs 51）')

axes[-1].set_xlabel('交易日')

# 计算与真实趋势的 MSE
from sklearn.metrics import mean_squared_error
valid_range = slice(50, -50)  # 排除边界区域
mse_sma = mean_squared_error(trend[valid_range], trend_sma[valid_range])
mse_firwin = mean_squared_error(trend[valid_range], trend_firwin[valid_range])
mse_remez = mean_squared_error(trend[valid_range], trend_remez[valid_range])

print(f"""
  三种方法提取趋势的 MSE（排除边界50点）：
    SMA-20:         {mse_sma:.4f}
    firwin (n=51):  {mse_firwin:.4f}
    remez (n=31):   {mse_remez:.4f}

  → remez 只用 31 阶就达到了和 firwin 51 阶类似的精度！
""")

fig.suptitle('SMA vs firwin vs remez — 趋势提取三方法对比',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch04_fig5_trend_extraction.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch04_fig5_trend_extraction.png")

# ============================================================
# Part 6: 电力实战 — remez 多带滤波器分离负荷分量
# ============================================================
print("\n" + "=" * 60)
print("Part 6: 电力实战 — 多带滤波器分离负荷分量")
print("=" * 60)

# 模拟 14 天电力负荷（15分钟采样）
np.random.seed(77)
points_per_hour = 4
fs_load = points_per_hour
hours = 24 * 14
N_load = hours * points_per_hour
t_load = np.arange(N_load) / fs_load  # 单位：小时

# 多个频率成分
# 14天周期（极低频趋势）
ultra_low = 0.1 * np.sin(2 * np.pi * t_load / (14 * 24))
# 24h 日周期
daily = 0.3 * np.sin(2 * np.pi * t_load / 24 - np.pi/2)
# 12h 半日周期
half_day = 0.08 * np.sin(2 * np.pi * t_load / 12)
# 高频噪声
noise = 0.03 * np.random.randn(N_load)

load = ultra_low + daily + half_day + noise

# 多带滤波器：分三带提取
# 带1 (0~0.02 次/小时 = 周期>50h)：超低频趋势 — gain=1
# 带2 (0.02~0.06 次/小时 = 周期17-50h)：日周期区域 — gain=0
# 带3 (0.06~2 次/小时 = 周期>0.5h)：高频 — gain=1
b_load_multi = remez(101,
                     bands=[0, 0.02, 0.025, 0.06, 0.07, 2],
                     desired=[1, 0, 1],
                     weight=[1, 5, 1],
                     fs=fs_load)

# 应用
load_filtered = filtfilt(b_load_multi, [1], load)

# 可视化
fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)

plot_hours = 4 * 24  # 显示前 4 天
t_plot = t_load[:plot_hours * points_per_hour]

axes[0].plot(t_plot, load[:len(t_plot)], linewidth=0.4, color='gray', alpha=0.7)
axes[0].plot(t_plot, ultra_low[:len(t_plot)] + daily[:len(t_plot)],
             linewidth=0.8, color='black', linestyle='--', alpha=0.6,
             label='趋势+日周期 (真值)')
axes[0].set_ylabel('负荷')
axes[0].set_title('原始负荷 — 含超低频趋势+日周期+半日周期+噪声')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_plot, load[:len(t_plot)], linewidth=0.3, color='gray', alpha=0.4)
axes[1].plot(t_plot, load_filtered[:len(t_plot)], linewidth=1.2, color='steelblue',
             label='多带滤波后 (去除日周期区域)')
axes[1].plot(t_plot, (ultra_low + noise)[:len(t_plot)],
             linewidth=0.6, color='black', linestyle='--', alpha=0.5,
             label='趋势+噪声 (目标)')
axes[1].set_ylabel('负荷')
axes[1].set_title('多带滤波器输出 — 保留趋势+高频噪声，去除日周期')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

# 分离出的日周期成分
daily_component = load - load_filtered
axes[2].plot(t_plot, daily_component[:len(t_plot)], linewidth=1.2, color='darkorange',
             label='分离出的日周期成分')
axes[2].plot(t_plot, (daily + half_day)[:len(t_plot)],
             linewidth=0.5, color='black', linestyle='--', alpha=0.6,
             label='真实日周期+半日周期')
axes[2].axhline(y=0, color='gray', linewidth=0.5)
axes[2].set_xlabel('时间 (小时)')
axes[2].set_ylabel('负荷偏离')
axes[2].set_title('剩余成分 (= 原始 - 滤波后) → 主要为日周期')
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

fig.suptitle('电力负荷多带滤波 — remez 一站式分离不同频率成分',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch04_fig6_power_multiband.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch04_fig6_power_multiband.png")

# ============================================================
# Part 7: OptimalFilterDesigner 工具类
# ============================================================
print("\n" + "=" * 60)
print("Part 7: OptimalFilterDesigner — 可复用工具类")
print("=" * 60)


class OptimalFilterDesigner:
    """
    最优 FIR 滤波器设计工具类 — 封装 remez 和 firwin 的统一接口。

    设计理念：
      - 用"周期"描述截止（而非频率）— 更符合金融/电力分析习惯
      - 自动在 firwin 和 remez 之间选择
      - 统一分析接口

    使用：
        ofd = OptimalFilterDesigner(fs=252)
        b = ofd.lowpass(period=20, numtaps=51, method='remez')
        y = ofd.apply(b, signal)
        ofd.analyze(b)
    """

    def __init__(self, fs=1.0):
        self.fs = fs
        self.nyq = fs / 2

    def _p2f(self, period):
        """周期 → 频率"""
        return self.fs / period if period > 0 else 0

    def _f2p(self, freq):
        """频率 → 周期"""
        return self.fs / freq if freq > 0 else np.inf

    def lowpass(self, period, numtaps=51, method='remez',
                transition_ratio=0.5, weight=None):
        """
        低通滤波器 — 保留周期 > period 的成分。

        Parameters
        ----------
        period : float
            截止周期（如 20 表示保留 20 天以上的趋势）
        method : str
            'remez' 或 'firwin'
        transition_ratio : float
            过渡带宽度占截止频率的比例（仅 remez）
            0.3 = 过渡带窄（需要更多阶数）
            0.7 = 过渡带宽（阶数需求低）
        """
        cutoff = self._p2f(period)
        if method == 'remez':
            trans_width = cutoff * transition_ratio
            bands = [0, cutoff - trans_width/2,
                     cutoff + trans_width/2, self.nyq]
            desired = [1, 0]
            if weight is None:
                weight = [1, 5]
            return remez(numtaps, bands, desired, weight=weight, fs=self.fs)
        else:
            return firwin(numtaps, cutoff, window='hamming', fs=self.fs)

    def highpass(self, period, numtaps=51, method='remez', **kwargs):
        """高通滤波器 — 保留周期 < period 的成分"""
        cutoff = self._p2f(period)
        if method == 'remez':
            trans_width = cutoff * kwargs.get('transition_ratio', 0.5)
            bands = [0, cutoff - trans_width/2,
                     cutoff + trans_width/2, self.nyq]
            desired = [0, 1]
            weight = kwargs.get('weight', [5, 1])
            return remez(numtaps, bands, desired, weight=weight, fs=self.fs)
        else:
            return firwin(numtaps, cutoff, window='hamming',
                         pass_zero=False, fs=self.fs)

    def bandpass(self, period_low, period_high, numtaps=51,
                 method='remez', **kwargs):
        """带通滤波器 — 保留 period_low ~ period_high 的成分"""
        f_low = self._p2f(period_high)
        f_high = self._p2f(period_low)
        if method == 'remez':
            ratio = kwargs.get('transition_ratio', 0.3)
            tw_low = f_low * ratio
            tw_high = f_high * ratio
            bands = [0, f_low - tw_low,
                     f_low + tw_low, f_high - tw_high,
                     f_high + tw_high, self.nyq]
            desired = [0, 1, 0]
            weight = kwargs.get('weight', [5, 1, 5])
            return remez(numtaps, bands, desired, weight=weight, fs=self.fs)
        else:
            return firwin(numtaps, [f_low, f_high], window='hamming',
                         pass_zero=False, fs=self.fs)

    @staticmethod
    def apply(b, x, zero_phase=True):
        """应用滤波器"""
        if zero_phase:
            return filtfilt(b, [1], x)
        return lfilter(b, [1], x)

    def analyze(self, b, title='Filter'):
        """绘制滤波器综合分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 7))

        # 频率响应
        w, h = freqz(b, worN=2048, fs=self.fs)
        h_db = 20 * np.log10(np.maximum(np.abs(h), 1e-15))
        axes[0, 0].plot(w, h_db, linewidth=1.2, color='steelblue')
        axes[0, 0].axhline(y=0, color='gray', linewidth=0.5)
        axes[0, 0].axhline(y=-3, color='red', linewidth=0.5, linestyle='--', alpha=0.6)
        axes[0, 0].axhline(y=-40, color='gray', linewidth=0.5, linestyle=':', alpha=0.6)
        axes[0, 0].set_xlabel('频率')
        axes[0, 0].set_ylabel('幅度 (dB)')
        axes[0, 0].set_title(f'{title} — 频率响应')
        axes[0, 0].set_ylim(-80, 5)
        axes[0, 0].grid(True, alpha=0.3)

        # 脉冲响应
        axes[0, 1].stem(np.arange(len(b)), b, linefmt='steelblue-',
                       markerfmt='steelblueo', basefmt='gray')
        axes[0, 1].set_xlabel('样本序号')
        axes[0, 1].set_ylabel('系数')
        axes[0, 1].set_title('脉冲响应 (滤波器系数)')
        axes[0, 1].grid(True, alpha=0.3)

        # 群延迟
        freqs, gd = group_delay((b, 1), fs=self.fs)
        axes[1, 0].plot(freqs, gd, linewidth=1.2, color='darkorange')
        axes[1, 0].axhline(y=(len(b)-1)//2, color='red', linewidth=0.8,
                          linestyle='--', alpha=0.7)
        axes[1, 0].set_xlabel('频率')
        axes[1, 0].set_ylabel('群延迟 (样本)')
        axes[1, 0].set_title(f'群延迟 (均值≈{np.mean(gd):.1f}样本)')
        axes[1, 0].grid(True, alpha=0.3)

        # 相位响应
        phase = np.unwrap(np.angle(h))
        axes[1, 1].plot(w, phase, linewidth=1.2, color='steelblue')
        axes[1, 1].set_xlabel('频率')
        axes[1, 1].set_ylabel('相位 (rad)')
        axes[1, 1].set_title('相位响应')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


# 演示
ofd = OptimalFilterDesigner(fs=252)
b_demo = ofd.lowpass(period=20, numtaps=31, method='remez',
                     transition_ratio=0.4, weight=[1, 10])

fig = ofd.analyze(b_demo, title='remez 低通 (周期>20天, n=31)')
plt.savefig('code/ch04_fig7_designer_demo.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch04_fig7_designer_demo.png")
print("  OptimalFilterDesigner 类已封装完毕。")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("第4章运行完毕！")
print("=" * 60)
print(f"""
📋 本章产出：

  笔记：doc/04_FIR滤波器设计_Remez最优法.md
  代码：code/ch04_FIR滤波器设计_Remez最优法.py
  图片：
    ch04_fig1_remez_vs_firwin.png        — 同阶数 remez vs firwin 全面 PK
    ch04_fig2_weight_sweep.png           — weight 参数扫参
    ch04_fig3_multiband.png              — 三带滤波器（firwin 做不到）
    ch04_fig4_minimum_phase.png          — 最小相位 FIR 降低延迟
    ch04_fig5_trend_extraction.png       — SMA vs firwin vs remez 趋势提取对比
    ch04_fig6_power_multiband.png        — 多带滤波器分离电力负荷分量
    ch04_fig7_designer_demo.png          — OptimalFilterDesigner 工具类

🎯 核心收获：
  1. remez = 等波纹最优 → 同等阶数下过渡带最窄
  2. bands/desired/weight 三个参数精确控制频率响应
  3. firwin = 简单快速, remez = 阶数效率最高
  4. 多带滤波器必须用 remez（firwin 只支持标准四种）
  5. 最小相位 FIR 用相位线性换延迟降低
  6. weight=[1,10] → 阻带精度是通带的10倍（金融趋势提取推荐）

📖 下一站：第5章 — IIR 滤波器设计
  → 以极低阶数实现陡峭过渡 — 嵌入式/实时系统的首选
""")
