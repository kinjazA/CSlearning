#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第10章 · LTI 系统分析与 Hilbert 变换 — 配套代码
====================================================
Part A: LTI 系统 — 传递函数 / 零极点 / 冲击响应 / Bode图
Part B: Hilbert — 解析信号 / 包络 / 瞬时频率

演示：
  1. LTI 三种表示形式及互相转换
  2. 冲击响应 & 阶跃响应
  3. Bode 图 & 频率响应
  4. 串联 / 并联 / 反馈
  5. Hilbert 包络提取 — 与峰值包络对比
  6. 瞬时频率 — chirp 信号的频率追踪

运行方式：
  python code/ch10_LTI系统与Hilbert变换.py

依赖：
  pip install numpy scipy matplotlib
"""

import numpy as np
from scipy import signal
from scipy.signal import (TransferFunction, ZerosPolesGain,
                          tf2zpk, zpk2tf, tf2ss,
                          impulse, step, bode, freqresp,
                          series, parallel, feedback,
                          hilbert)
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.dpi': 120, 'font.size': 9,
    'axes.titlesize': 11, 'axes.labelsize': 9,
})
print("=" * 60)
print("第10章 · LTI 系统分析与 Hilbert 变换")
print("=" * 60)

# ============================================================
# Part A1: LTI 三种表示 + 互相转换
# ============================================================
print("\n" + "=" * 60)
print("Part A1: LTI 系统的三种表示")
print("=" * 60)

# 以 Butterworth 3阶低通为例
fs = 100
b, a = signal.butter(3, 10, btype='low', fs=fs)
tf = TransferFunction(b, a, dt=1/fs)  # dt 表示离散系统

# 转换为零极点增益
z, p, k = tf2zpk(b, a)

# 再转回传递函数
b2, a2 = zpk2tf(z, p, k)

print(f"  b (前馈系数):  {b}")
print(f"  a (反馈系数):  {a}")
print(f"  z (零点):      {z}")
print(f"  p (极点):      {p}")
print(f"  k (增益):      {k:.4f}")
print(f"  极点模 |p|:    {np.abs(p)}  → {'全部<1，稳定' if np.all(np.abs(p)<1) else '有极点>=1，不稳定！'}")

# ============================================================
# Part A2: 冲击响应 & 阶跃响应
# ============================================================
print("\n" + "=" * 60)
print("Part A2: 冲击响应 & 阶跃响应")
print("=" * 60)

# 用三个不同阻尼比的二阶系统演示
# 二阶传递函数: H(s) = ωn² / (s² + 2ζωn s + ωn²)
# ζ = 阻尼比：<1 欠阻尼（振荡衰减），=1 临界阻尼，>1 过阻尼

damping_ratios = [0.2, 0.7, 1.5]
omega_n = 2 * np.pi * 5  # 自然频率 5 Hz

fig, axes = plt.subplots(2, 2, figsize=(16, 8))

for zeta in damping_ratios:
    # 连续系统传递函数：模拟二阶低通
    num = [omega_n**2]
    den = [1, 2 * zeta * omega_n, omega_n**2]
    tf_sys = TransferFunction(num, den)

    # 冲击响应
    t_imp, y_imp = impulse(tf_sys, T=np.linspace(0, 1.5, 500))
    axes[0, 0].plot(t_imp, y_imp, linewidth=1.2,
                    label=f'ζ={zeta}')

    # 阶跃响应
    t_step, y_step = step(tf_sys, T=np.linspace(0, 1.5, 500))
    axes[0, 1].plot(t_step, y_step, linewidth=1.2,
                    label=f'ζ={zeta}')

    # Bode 图（幅度）
    w, mag, phase = bode(tf_sys, w=np.logspace(0, 2, 500))
    axes[1, 0].semilogx(w, mag, linewidth=1.2, label=f'ζ={zeta}')

    # 零极点图
    z_s, p_s, k_s = tf2zpk(num, den)
    axes[1, 1].plot(np.real(p_s), np.imag(p_s), 'x', markersize=12,
                    markeredgewidth=2, label=f'ζ={zeta} 极点')

axes[0, 0].set_xlabel('时间 (s)'); axes[0, 0].set_ylabel('响应')
axes[0, 0].set_title('冲击响应 — ζ<1 振荡衰减，ζ>1 单调衰减')
axes[0, 0].legend(fontsize=8); axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].set_xlabel('时间 (s)'); axes[0, 1].set_ylabel('响应')
axes[0, 1].axhline(y=1, color='gray', linewidth=0.5)
axes[0, 1].set_title('阶跃响应 — 输入从0跳到1，输出如何跟踪')
axes[0, 1].legend(fontsize=8); axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].set_xlabel('频率 (rad/s)'); axes[1, 0].set_ylabel('幅度 (dB)')
axes[1, 0].axhline(y=-3, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
axes[1, 0].set_title('Bode 图（幅度）— ζ 大=衰减平缓，ζ 小=截止附近有共振峰')
axes[1, 0].legend(fontsize=8); axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].axvline(x=0, color='gray', linewidth=0.3)
axes[1, 1].axhline(y=0, color='gray', linewidth=0.3)
axes[1, 1].set_xlabel('实部'); axes[1, 1].set_ylabel('虚部')
axes[1, 1].set_title('零极点图 — ζ<1 极点在左半平面（稳定），ζ 越小越靠近虚轴')
axes[1, 1].legend(fontsize=8); axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_aspect('equal')

fig.suptitle('LTI 系统分析 — 阻尼比 ζ 如何影响系统行为',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch10_fig1_lti_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch10_fig1_lti_analysis.png")

# ============================================================
# Part A3: 串联 / 并联 / 反馈
# ============================================================
print("\n" + "=" * 60)
print("Part A3: 串联 + 并联 + 反馈")
print("=" * 60)

# 两个子系统
b1, a1 = signal.butter(2, 8, btype='low', fs=fs)
b2, a2 = signal.butter(2, 15, btype='low', fs=fs)
tf1 = TransferFunction(b1, a1, dt=1/fs)
tf2 = TransferFunction(b2, a2, dt=1/fs)

# 串联、并联、反馈
tf_series   = series(tf1, tf2)
tf_parallel = parallel(tf1, tf2)
tf_feedback = feedback(tf1, tf2)

# 频率响应对比
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

for ax, (sys, desc, color) in zip(axes,
    [(tf_series, '串联 (tf1 → tf2) — 两次滤波级联，衰减加倍', 'steelblue'),
     (tf_parallel, '并联 (tf1 + tf2) — 两条路径叠加', 'darkorange'),
     (tf_feedback, '反馈 — 输出的一部分返回输入', 'crimson')]):
    w, h = freqresp(sys, w=np.logspace(-1, 1.7, 500))
    h_db = 20 * np.log10(np.maximum(np.abs(h), 1e-15))
    ax.semilogx(w, h_db, linewidth=1.2, color=color)
    ax.axhline(y=-3, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.set_ylabel('幅度 (dB)')
    ax.set_title(desc)
    ax.set_ylim(-60, 5)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('频率 (rad/s)')
fig.suptitle('系统连接方式 — 串联/并联/反馈对频率响应的影响',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch10_fig2_interconnection.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch10_fig2_interconnection.png")

# ============================================================
# Part B1: Hilbert 包络 — 与峰值包络对比
# ============================================================
print("\n" + "=" * 60)
print("Part B1: Hilbert 包络 vs 峰值包络")
print("=" * 60)

# 构造一个幅度调制的信号
np.random.seed(42)
t_env = np.linspace(0, 10, 2000)
fs_env = 200
# 载波：5 Hz 正弦
carrier = np.sin(2 * np.pi * 5 * t_env)
# 调制：幅度缓慢变化 (0.5 Hz) + 衰减趋势
amplitude_mod = 1.5 + 0.8 * np.sin(2 * np.pi * 0.5 * t_env) - 0.03 * t_env
modulated = amplitude_mod * carrier + 0.1 * np.random.randn(len(t_env))

# 方法1：Hilbert 包络
analytic = hilbert(modulated)
hilbert_envelope = np.abs(analytic)

# 方法2：峰值包络（第9章的方法）
peaks_env, _ = signal.find_peaks(modulated, distance=15)
peak_envelope = np.interp(t_env, t_env[peaks_env], modulated[peaks_env])

# 图
fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

p = slice(0, 600)
axes[0].plot(t_env[p], modulated[p], linewidth=0.3, color='steelblue',
             alpha=0.7, label='信号')
axes[0].plot(t_env[p], amplitude_mod[p], linewidth=0.6, color='black',
             linestyle='--', alpha=0.6, label='真实包络 (已知)')
axes[0].plot(t_env[p], hilbert_envelope[p], linewidth=1.5, color='crimson',
             alpha=0.8, label='Hilbert 包络')
axes[0].set_ylabel('幅值')
axes[0].set_title('Hilbert 包络 — 光滑连续，完美贴合真实包络')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_env[p], modulated[p], linewidth=0.3, color='steelblue',
             alpha=0.7, label='信号')
axes[1].plot(t_env[p], amplitude_mod[p], linewidth=0.6, color='black',
             linestyle='--', alpha=0.6, label='真实包络')
axes[1].plot(t_env[peaks_env], modulated[peaks_env], 'o', markersize=4,
             color='darkorange', alpha=0.7, label='检测到的峰')
axes[1].plot(t_env[p], peak_envelope[p], linewidth=1.5, color='darkorange',
             alpha=0.8, label='峰值插值包络')
axes[1].set_xlabel('时间 (s)')
axes[1].set_ylabel('幅值')
axes[1].set_title('峰值插值包络 — 依赖峰检测质量，峰位置之间的区域靠插值')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

fig.suptitle('Hilbert 包络 vs 峰值插值包络 — Hilbert 更光滑、更精确',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch10_fig3_hilbert_envelope.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch10_fig3_hilbert_envelope.png")

# ============================================================
# Part B2: 瞬时频率 — chirp 信号追踪
# ============================================================
print("\n" + "=" * 60)
print("Part B2: 瞬时频率 — 追踪 chirp 的扫频轨迹")
print("=" * 60)

# chirp 信号：频率从 10 Hz 线性扫到 100 Hz
t_if = np.linspace(0, 3, 3000)
fs_if = 1000
chirp_sig = signal.chirp(t_if, f0=10, t1=3, f1=100, method='linear')
chirp_sig += 0.05 * np.random.randn(len(t_if))

# Hilbert 瞬时频率
analytic_if = hilbert(chirp_sig)
phase_if = np.unwrap(np.angle(analytic_if))
inst_freq = np.diff(phase_if) / (2 * np.pi * (1/fs_if))  # Hz

# 真实频率（用于对比）
true_freq = 10 + (100 - 10) * t_if / 3

# 图
fig, axes = plt.subplots(3, 1, figsize=(16, 10))

# 时域
axes[0].plot(t_if[:400], chirp_sig[:400], linewidth=0.4, color='steelblue')
axes[0].set_ylabel('幅值')
axes[0].set_title('Chirp 信号时域（前 0.4s）— 频率从 10Hz 扫到 100Hz')
axes[0].grid(True, alpha=0.3)

# 瞬时频率 vs 真实频率
axes[1].plot(t_if[1:], inst_freq, linewidth=0.3, color='steelblue', alpha=0.8,
             label='Hilbert 瞬时频率')
axes[1].plot(t_if, true_freq, linewidth=1.5, color='crimson', linestyle='--',
             alpha=0.7, label='真实频率 (10→100Hz)')
axes[1].set_ylabel('频率 (Hz)')
axes[1].set_title('瞬时频率追踪 — 与真实扫频轨迹几乎完全重合')
axes[1].legend(fontsize=9)
axes[1].set_ylim(0, 120)
axes[1].grid(True, alpha=0.3)

# 误差
freq_error = inst_freq - true_freq[1:]
axes[2].plot(t_if[1:], freq_error, linewidth=0.5, color='crimson')
axes[2].axhline(y=0, color='gray', linewidth=0.5)
axes[2].set_xlabel('时间 (s)')
axes[2].set_ylabel('误差 (Hz)')
axes[2].set_title(f'频率追踪误差 — RMS = {np.sqrt(np.mean(freq_error**2)):.2f} Hz')
axes[2].set_ylim(-10, 10)
axes[2].grid(True, alpha=0.3)

fig.suptitle('Hilbert 瞬时频率 — 精确追踪 chirp 信号的频率变化',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch10_fig4_instantaneous_freq.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch10_fig4_instantaneous_freq.png")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("第10章运行完毕！")
print("=" * 60)
print(f"""
📋 本章产出：

  笔记：doc/10_LTI系统与Hilbert变换.md
  代码：code/ch10_LTI系统与Hilbert变换.py
  图片：
    ch10_fig1_lti_analysis.png         — 阻尼比对冲击/阶跃/Bode/零极点的影响
    ch10_fig2_interconnection.png      — 串联/并联/反馈 频率响应对比
    ch10_fig3_hilbert_envelope.png     — Hilbert 包络 vs 峰值插值包络
    ch10_fig4_instantaneous_freq.png   — 瞬时频率追踪 chirp 信号

🎯 核心收获：
  Part A — LTI 系统
    1. 三种等价表示：(b,a)传递函数 / (z,p,k)零极点 / (A,B,C,D)状态空间
    2. 冲击响应=系统的"指纹"，阶跃响应=系统的"反应速度"
    3. 零极点图：极点在单位圆(离散)或左半平面(连续)内=稳定
    4. series/parallel/feedback = 搭建复杂系统的积木

  Part B — Hilbert 变换
    5. 解析信号 = 实信号 + j×Hilbert变换 → 提取包络和瞬时频率
    6. Hilbert 包络：光滑连续，数学严格，优于峰值插值
    7. 瞬时频率：相位求导 → 每个时刻的主频率
    8. unwrap：解开相位缠绕，让相位连续

📖 下一站：第11章 — 重采样与预处理管线
  → 降采样/升采样、抗混叠、去趋势、归一化 → 构建可复用的预处理Pipeline
""")
