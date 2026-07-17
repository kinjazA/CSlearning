#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第5章 · IIR 滤波器设计与应用 — 配套代码
=============================================
核心主题：以极低阶数实现陡峭过渡——实时系统/嵌入式的首选

演示内容：
  1. 五种经典 IIR 原型同阶数横向对比 (Butterworth / Cheby1 / Cheby2 / Elliptic / Bessel)
  2. Elliptic 的 rp/rs 参数扫参
  3. Butterworth 阶数扫参（阶数越大 → 过渡带越窄）
  4. 零极点图 — 稳定性的可视化判别
  5. SOS vs (b,a) — 为什么高阶 IIR 必须用二阶节
  6. 股价滤波实战：IIR vs FIR 延迟对比
  7. 实时滤波 vs 零相位滤波 (sosfilt vs sosfiltfilt)
  8. 封装 IIRFilterDesigner 工具类

运行方式：
  python code/ch05_IIR滤波器设计.py

依赖：
  pip install numpy scipy matplotlib
"""

import numpy as np
from scipy import signal
from scipy.signal import (butter, cheby1, cheby2, ellip, bessel,
                          iirdesign, iirfilter,
                          freqz, group_delay, sosfreqz,
                          lfilter, filtfilt, sosfilt, sosfiltfilt,
                          tf2sos, tf2zpk)
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
print("第5章 · IIR 滤波器设计与应用")
print("=" * 60)

fs = 100  # 采样率 100 Hz（本实验用）

# ============================================================
# Part 1: 五种 IIR 原型同阶数横向对比
# ============================================================
print("\n" + "=" * 60)
print("Part 1: 五种 IIR 原型 — 同阶数下谁更陡？谁更平？")
print("=" * 60)

N_order = 4          # 低通滤波器阶数
cutoff = 10          # 截止频率 10 Hz

# 设计五种原型的低通滤波器（都转 SOS 以确保稳定）
sos_butter = butter(N_order, cutoff, btype='low', fs=fs, output='sos')
sos_cheby1 = cheby1(N_order, rp=0.5, Wn=cutoff, btype='low', fs=fs, output='sos')
sos_cheby2 = cheby2(N_order, rs=40, Wn=cutoff, btype='low', fs=fs, output='sos')
sos_ellip  = ellip(N_order, rp=0.5, rs=40, Wn=cutoff, btype='low', fs=fs, output='sos')
sos_bessel = bessel(N_order, cutoff, btype='low', fs=fs, output='sos')

filters = {
    'Butterworth':       (sos_butter, 'steelblue'),
    'Chebyshev I':       (sos_cheby1, 'darkorange'),
    'Chebyshev II':      (sos_cheby2, 'green'),
    'Elliptic (Cauer)':  (sos_ellip,  'crimson'),
    'Bessel':            (sos_bessel, 'purple'),
}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 子图1：频率响应 — 全览
ax = axes[0, 0]
for name, (sos, color) in filters.items():
    w, h = sosfreqz(sos, worN=2048, fs=fs)
    h_db = 20 * np.log10(np.maximum(np.abs(h), 1e-15))
    ax.plot(w, h_db, linewidth=1.2, color=color, alpha=0.85, label=name)
ax.axhline(y=0, color='gray', linewidth=0.5)
ax.axhline(y=-3, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
ax.axvline(x=10, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
ax.set_xlabel('频率 (Hz)')
ax.set_ylabel('幅度 (dB)')
ax.set_title(f'频率响应全览 (N={N_order}, cutoff=10Hz)')
ax.set_ylim(-80, 5)
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# 子图2：频率响应 — 过渡带放大
ax = axes[0, 1]
for name, (sos, color) in filters.items():
    w, h = sosfreqz(sos, worN=2048, fs=fs)
    h_db = 20 * np.log10(np.maximum(np.abs(h), 1e-15))
    ax.plot(w, h_db, linewidth=1.5, color=color, alpha=0.85, label=name)
ax.axhline(y=-3, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
ax.axvline(x=10, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)
ax.set_xlabel('频率 (Hz)')
ax.set_ylabel('幅度 (dB)')
ax.set_title('过渡带放大')
ax.set_xlim(5, 25)
ax.set_ylim(-50, 2)
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# 子图3：频率响应 — 通带放大（看纹波/平坦度）
ax = axes[0, 2]
for name, (sos, color) in filters.items():
    w, h = sosfreqz(sos, worN=2048, fs=fs)
    h_db = 20 * np.log10(np.maximum(np.abs(h), 1e-15))
    ax.plot(w, h_db, linewidth=1.2, color=color, alpha=0.85, label=name)
ax.axhline(y=0, color='gray', linewidth=0.5)
ax.axhline(y=-0.5, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
ax.axhline(y=-3, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
ax.set_xlabel('频率 (Hz)')
ax.set_ylabel('幅度 (dB)')
ax.set_title(f'通带放大 (纹波可见)')
ax.set_xlim(0, 12)
ax.set_ylim(-4, 1)
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# 子图4-6：群延迟对比（三种原型）
delay_plots = [
    (axes[1, 0], ['Butterworth', 'Bessel'], 'Butterworth vs Bessel — 群延迟'),
    (axes[1, 1], ['Chebyshev I', 'Chebyshev II'], 'Chebyshev I vs II — 群延迟'),
    (axes[1, 2], ['Elliptic (Cauer)'], 'Elliptic — 群延迟（变化最剧烈）'),
]
for ax, keys, title in delay_plots:
    for name in keys:
        sos, color = filters[name]
        freqs, gd = group_delay(sos, fs=fs)
        ax.plot(freqs, gd, linewidth=1.2, color=color, alpha=0.85, label=name)
    ax.set_xlabel('频率 (Hz)')
    ax.set_ylabel('群延迟 (样本)')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle(f'五种 IIR 原型同阶数对比 (N={N_order}) — 没有完美的滤波器，只有合适的权衡',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch05_fig1_iir_prototypes.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch05_fig1_iir_prototypes.png")

# 打印过渡带宽度对比
print(f"\n  同阶数 (N={N_order}) 低通滤波器的过渡带对比：")
print(f"  {'原型':<18s} {'-3dB (Hz)':<12s} {'-40dB (Hz)':<12s} {'过渡带宽度':<12s}")
print(f"  {'-'*50}")
for name, (sos, _) in filters.items():
    w, h = sosfreqz(sos, worN=4096, fs=fs)
    h_db = 20 * np.log10(np.maximum(np.abs(h), 1e-15))
    idx_3db  = np.argmin(np.abs(h_db + 3))
    idx_40db = np.argmin(np.abs(h_db + 40))
    f_3db  = w[idx_3db]
    f_40db = w[idx_40db] if idx_40db < len(w) else fs/2
    print(f"  {name:<18s} {f_3db:<12.2f} {f_40db:<12.2f} {f_40db-f_3db:<12.2f}")
print(f"\n  → Elliptic 过渡带最窄，Bessel 最宽。")
print(f"  → 但这只是'频率选择性'维度。注意看上图群延迟的差别！")

# ============================================================
# Part 2: Elliptic rp/rs 扫参
# ============================================================
print("\n" + "=" * 60)
print("Part 2: Elliptic — rp 和 rs 参数如何影响过渡带")
print("=" * 60)

rp_values = [0.1, 0.5, 1.0, 3.0]
rs_values = [20, 40, 60, 80]

fig, axes = plt.subplots(2, 2, figsize=(16, 9))

# rp 扫参（固定 rs=40）
ax = axes[0, 0]
for rp in rp_values:
    sos = ellip(N_order, rp=rp, rs=40, Wn=cutoff, btype='low', fs=fs, output='sos')
    w, h = sosfreqz(sos, worN=2048, fs=fs)
    h_db = 20 * np.log10(np.maximum(np.abs(h), 1e-15))
    ax.plot(w, h_db, linewidth=1.2, alpha=0.85,
            label=f'rp={rp} dB')
ax.axhline(y=0, color='gray', linewidth=0.5)
ax.axvline(x=cutoff, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
ax.set_xlabel('频率 (Hz)'); ax.set_ylabel('幅度 (dB)')
ax.set_title('rp 参数扫参 (rs=40 固定) — rp越大=通带纹波越大=过渡带越窄')
ax.set_xlim(0, 25); ax.set_ylim(-80, 2)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# rs 扫参（固定 rp=0.5）
ax = axes[0, 1]
for rs in rs_values:
    sos = ellip(N_order, rp=0.5, rs=rs, Wn=cutoff, btype='low', fs=fs, output='sos')
    w, h = sosfreqz(sos, worN=2048, fs=fs)
    h_db = 20 * np.log10(np.maximum(np.abs(h), 1e-15))
    ax.plot(w, h_db, linewidth=1.2, alpha=0.85,
            label=f'rs={rs} dB')
ax.axhline(y=0, color='gray', linewidth=0.5)
ax.axhline(y=-40, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
ax.axvline(x=cutoff, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
ax.set_xlabel('频率 (Hz)'); ax.set_ylabel('幅度 (dB)')
ax.set_title('rs 参数扫参 (rp=0.5 固定) — rs越大=阻带越干净=过渡带越宽')
ax.set_xlim(0, 25); ax.set_ylim(-80, 2)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 通带放大
ax = axes[1, 0]
for rp in rp_values:
    sos = ellip(N_order, rp=rp, rs=40, Wn=cutoff, btype='low', fs=fs, output='sos')
    w, h = sosfreqz(sos, worN=2048, fs=fs)
    h_db = 20 * np.log10(np.maximum(np.abs(h), 1e-15))
    ax.plot(w, h_db, linewidth=1.2, alpha=0.85, label=f'rp={rp}')
ax.axhline(y=0, color='gray', linewidth=0.5)
for rp in rp_values:
    ax.axhline(y=-rp, color='red', linewidth=0.5, linestyle=':', alpha=0.3)
ax.set_xlabel('频率 (Hz)'); ax.set_ylabel('幅度 (dB)')
ax.set_title('通带放大 — rp 控制通带纹波的幅度')
ax.set_xlim(0, 10); ax.set_ylim(-4, 1)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 群延迟 — rp/rs 对相位的影响
ax = axes[1, 1]
for rp, rs in [(0.1, 40), (0.5, 40), (0.5, 60), (3.0, 80)]:
    sos = ellip(N_order, rp=rp, rs=rs, Wn=cutoff, btype='low', fs=fs, output='sos')
    freqs, gd = group_delay(sos, fs=fs)
    ax.plot(freqs, gd, linewidth=1.2, alpha=0.85,
            label=f'rp={rp}, rs={rs}')
ax.set_xlabel('频率 (Hz)'); ax.set_ylabel('群延迟 (样本)')
ax.set_title('群延迟 — rp/rs 越大 → 群延迟越不平坦（相位失真越严重）')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

fig.suptitle(f'Elliptic 滤波器 rp/rs 参数详解 (N={N_order})', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch05_fig2_elliptic_sweep.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch05_fig2_elliptic_sweep.png")

# ============================================================
# Part 3: Butterworth 阶数扫参
# ============================================================
print("\n" + "=" * 60)
print("Part 3: Butterworth — 阶数越大，过渡带越窄")
print("=" * 60)

orders = [2, 4, 6, 8, 12]

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

for N in orders:
    sos = butter(N, cutoff, btype='low', fs=fs, output='sos')
    w, h = sosfreqz(sos, worN=2048, fs=fs)
    h_db = 20 * np.log10(np.maximum(np.abs(h), 1e-15))
    axes[0].plot(w, h_db, linewidth=1.2, alpha=0.85, label=f'N={N}')

    freqs, gd = group_delay(sos, fs=fs)
    axes[1].plot(freqs, gd, linewidth=1.2, alpha=0.85, label=f'N={N}')

axes[0].axhline(y=0, color='gray', linewidth=0.5)
axes[0].axhline(y=-3, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
axes[0].axvline(x=cutoff, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
axes[0].set_xlabel('频率 (Hz)')
axes[0].set_ylabel('幅度 (dB)')
axes[0].set_title('Butterworth — 阶数对频率响应的影响')
axes[0].set_ylim(-80, 5)
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('频率 (Hz)')
axes[1].set_ylabel('群延迟 (样本)')
axes[1].set_title('Butterworth — 阶数对群延迟的影响（阶数越高，延迟越大越不平坦）')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

fig.suptitle('Butterworth 阶数扫参 — IIR 只需 4-8 阶就足够陡峭',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch05_fig3_butterworth_orders.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch05_fig3_butterworth_orders.png")

# ============================================================
# Part 4: 零极点图 — IIR 稳定性的可视化判别
# ============================================================
print("\n" + "=" * 60)
print("Part 4: 零极点图 — 稳定性一目了然")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# 获取四种原型的 (b, a) 用于零极点分析
b_butter, a_butter = butter(N_order, cutoff, btype='low', fs=fs, output='ba')
b_cheby1, a_cheby1 = cheby1(N_order, rp=0.5, Wn=cutoff, btype='low', fs=fs, output='ba')
b_ellip, a_ellip = ellip(N_order, rp=0.5, rs=40, Wn=cutoff, btype='low', fs=fs, output='ba')
b_bessel, a_bessel = bessel(N_order, cutoff, btype='low', fs=fs, output='ba')

zp_data = [
    (axes[0, 0], b_butter, a_butter, 'Butterworth'),
    (axes[0, 1], b_cheby1, a_cheby1, 'Chebyshev I'),
    (axes[1, 0], b_ellip, a_ellip, 'Elliptic'),
    (axes[1, 1], b_bessel, a_bessel, 'Bessel'),
]

for ax, b, a, name in zp_data:
    z, p, k = tf2zpk(b, a)

    # 画单位圆
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=0.8, alpha=0.5)

    # 画零点（圆圈）
    ax.plot(np.real(z), np.imag(z), 'o', markersize=8, markerfacecolor='white',
            markeredgecolor='steelblue', markeredgewidth=1.5, label='Zeros (零点)')

    # 画极点（叉号）
    ax.plot(np.real(p), np.imag(p), 'x', markersize=10, color='crimson',
            markeredgewidth=2, label='Poles (极点)')

    # 判断稳定性
    is_stable = np.all(np.abs(p) < 1)
    status = '✓ 稳定' if is_stable else '✗ 不稳定！'
    color_status = 'green' if is_stable else 'red'

    ax.set_xlabel('实部'); ax.set_ylabel('虚部')
    ax.set_title(f'{name} — {status}', color=color_status, fontweight='bold')
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='gray', linewidth=0.3)
    ax.axvline(x=0, color='gray', linewidth=0.3)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle('零极点图 — 极点在单位圆内 = 稳定，极点在单位圆上或外 = 不稳定',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch05_fig4_pole_zero.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch05_fig4_pole_zero.png")
print("""
  零极点图解读：
    - 单位圆（黑线）：稳定性边界
    - 叉号（×）= 极点：在圆内 → 稳定，在圆上/外 → 不稳定
    - 圆圈（○）= 零点：决定滤波器对特定频率的"完全阻挡"位置
    - 所有四种原型在低阶时都是稳定的（极点在单位圆内）
""")

# ============================================================
# Part 5: SOS vs (b,a) — 数值精度对比
# ============================================================
print("\n" + "=" * 60)
print("Part 5: SOS vs (b,a) — 为什么高阶 IIR 必须用二阶节")
print("=" * 60)

# 用高阶 Butterworth 演示 (b,a) 格式的数值问题
N_high = 12  # 高阶

# 方式1：(b, a) 格式 — 可能有问题
b_ba, a_ba = butter(N_high, cutoff, btype='low', fs=fs, output='ba')

# 方式2：SOS 格式 — 数值稳定
sos_safe = butter(N_high, cutoff, btype='low', fs=fs, output='sos')

# 比较系数范围
print(f"\n  {N_high} 阶 Butterworth 的系数范围对比：")
print(f"  (b,a) 格式 — b 系数范围: [{np.min(np.abs(b_ba)):.2e}, {np.max(np.abs(b_ba)):.2e}]")
print(f"  (b,a) 格式 — a 系数范围: [{np.min(np.abs(a_ba)):.2e}, {np.max(np.abs(a_ba)):.2e}]")
for i, section in enumerate(sos_safe):
    print(f"  SOS 第{i+1}节 — b 范围: [{np.min(np.abs(section[:3])):.2e}, {np.max(np.abs(section[:3])):.2e}], "
          f"a 范围: [{np.min(np.abs(section[3:6])):.2e}, {np.max(np.abs(section[3:6])):.2e}]")

# 对比两者频率响应（应完全相同，因为数学上等价）
w_ba, h_ba = freqz(b_ba, a_ba, worN=2048, fs=fs)
w_sos, h_sos = sosfreqz(sos_safe, worN=2048, fs=fs)

fig, axes = plt.subplots(2, 1, figsize=(14, 7))

axes[0].plot(w_ba, 20*np.log10(np.maximum(np.abs(h_ba), 1e-15)),
             linewidth=1.5, color='steelblue', alpha=0.7, label='(b,a) 格式')
axes[0].plot(w_sos, 20*np.log10(np.maximum(np.abs(h_sos), 1e-15)),
             linewidth=1.0, color='crimson', alpha=0.5, linestyle='--', label='SOS 格式')
axes[0].set_xlabel('频率 (Hz)')
axes[0].set_ylabel('幅度 (dB)')
axes[0].set_title(f'频率响应 — (b,a) vs SOS (N={N_high}) — 理论上应完全相同')
axes[0].set_ylim(-80, 5)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# 系数分布对比
axes[1].stem(np.arange(len(b_ba)), np.abs(b_ba), linefmt='steelblue-',
             markerfmt='steelblueo', basefmt='gray', label="b (前馈)")
axes[1].stem(np.arange(len(a_ba)), np.abs(a_ba), linefmt='crimson-',
             markerfmt='crimsonx', basefmt='gray', label="a (反馈)")
axes[1].set_xlabel('系数序号')
axes[1].set_ylabel('系数的绝对值')
axes[1].set_title(f'(b,a) 系数分布 — 系数范围跨越 {np.log10(np.max(np.abs(a_ba))/np.min(np.abs(a_ba))):.0f} 个数量级')
axes[1].set_yscale('log')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

fig.suptitle('SOS vs (b,a) — 频率响应等价，但 SOS 在数值上安全得多',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch05_fig5_sos_vs_ba.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch05_fig5_sos_vs_ba.png")
print(f"""
  核心教训：
    - (b,a) 格式的 a 系数跨越约 {np.log10(np.max(np.abs(a_ba))/np.min(np.abs(a_ba))):.0f} 个数量级
    - 双精度浮点数只有约 15 位有效数字
    - 当数量级跨度 > 10 且 N > 8 时，(b,a) 格式有风险
    - SOS 每个子滤波器的系数范围仅跨越 1-2 个数量级 → 安全
    - 结论：永远用 output='sos' + sosfilt
""")

# ============================================================
# Part 6: 股价滤波实战 — IIR vs FIR 延迟对比
# ============================================================
print("\n" + "=" * 60)
print("Part 6: 股价滤波实战 — IIR 低延迟 vs FIR 高保真")
print("=" * 60)

# 构造含噪声的股价
np.random.seed(42)
fs_stock = 252  # 日度
N_days = 504    # 约2年
t_days = np.arange(N_days) / fs_stock
trend = 50 + 15 * t_days + 5 * np.sin(2 * np.pi * t_days * 2)
noise = 4 * np.random.randn(N_days)
price = trend + noise

# IIR: Butterworth 4阶低通（截止周期 25天）
cutoff_stock = fs_stock / 25
sos_iir = butter(4, cutoff_stock, btype='low', fs=fs_stock, output='sos')

# FIR: firwin 低通（需要类似过渡带的阶数）
# 要达到和4阶 IIR 类似的过渡带，FIR 约需 61 阶
b_fir = signal.firwin(61, cutoff=cutoff_stock, window='hamming', fs=fs_stock)

# 零相位滤波：IIR + filtfilt（补偿 IIR 的非线性相位）
price_iir_filtfilt = sosfiltfilt(sos_iir, price)
# 实时滤波：IIR + sosfilt（有延迟但可在线运行）
price_iir_realtime = sosfilt(sos_iir, price)
# FIR 零相位
price_fir = filtfilt(b_fir, [1], price)

# 可视化
fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

# 第一行：原始+真实趋势
axes[0].plot(price, linewidth=0.3, color='gray', alpha=0.5, label='原始股价')
axes[0].plot(trend, linewidth=0.8, color='black', linestyle='--',
             alpha=0.7, label='真实趋势')
axes[0].set_ylabel('价格')
axes[0].set_title('原始股价 & 真实趋势')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# 第二行：IIR vs FIR 零相位对比
axes[1].plot(trend, linewidth=0.8, color='black', linestyle='--', alpha=0.5,
             label='真实趋势')
axes[1].plot(price_iir_filtfilt, linewidth=1.2, color='crimson',
             label='IIR (Butterworth N=4) + filtfilt', alpha=0.8)
axes[1].plot(price_fir, linewidth=1.0, color='steelblue',
             label='FIR (numtaps=61) + filtfilt', alpha=0.6, linestyle='--')
axes[1].set_ylabel('价格')
axes[1].set_title('零相位滤波对比 — 4阶 IIR ≈ 61阶 FIR（注意延迟：0 vs 30天）')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

# 第三行：实时模式对比（暴露 IIR 的非线性相位）
axes[2].plot(trend, linewidth=0.8, color='black', linestyle='--', alpha=0.5,
             label='真实趋势')
axes[2].plot(price_iir_realtime, linewidth=1.0, color='darkorange',
             label='IIR 实时 (sosfilt, 有相位失真)', alpha=0.8)
axes[2].plot(price_iir_filtfilt, linewidth=0.8, color='crimson',
             label='IIR 零相位 (sosfiltfilt)', alpha=0.5, linestyle='--')
axes[2].set_xlabel('交易日')
axes[2].set_ylabel('价格')
axes[2].set_title('实时滤波 vs 零相位滤波 — sosfilt 有相位失真，sosfiltfilt 补偿了它')
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

fig.suptitle('IIR (4阶) vs FIR (61阶) — 股价趋势提取对比',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch05_fig6_stock_iir_vs_fir.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch05_fig6_stock_iir_vs_fir.png")

# 量化对比（排除边界）
valid = slice(60, -60)
mse_iir = np.mean((trend[valid] - price_iir_filtfilt[valid])**2)
mse_fir = np.mean((trend[valid] - price_fir[valid])**2)
print(f"""
  MSE 对比（排除边界60点）：
    IIR Butterworth (N=4) + filtfilt:  {mse_iir:.4f}
    FIR firwin (numtaps=61) + filtfilt: {mse_fir:.4f}
  → 性能相近，但 IIR 只用 4 阶 vs FIR 的 61 阶！
""")

# ============================================================
# Part 7: IIRFilterDesigner 工具类
# ============================================================
print("\n" + "=" * 60)
print("Part 7: IIRFilterDesigner — 可复用工具类")
print("=" * 60)


class IIRFilterDesigner:
    """
    IIR 滤波器设计与分析工具类。

    设计理念：
      - 默认转为 SOS 格式（安全第一）
      - 统一接口，底层可切换五种原型
      - 集成 freqz/group_delay 分析

    使用：
        iid = IIRFilterDesigner(fs=252)
        sos = iid.lowpass(cutoff_period=25, N=4, ftype='butter')
        y = iid.apply(sos, signal, zero_phase=True)
        iid.analyze(sos)
    """

    def __init__(self, fs=1.0):
        self.fs = fs
        self.nyq = fs / 2

    def _p2f(self, period):
        """周期 → 频率"""
        return self.fs / period if period > 0 else 0

    def _design(self, N, Wn, btype, ftype='butter', rp=0.5, rs=40):
        """内部统一设计接口，返回 SOS 格式"""
        design_funcs = {
            'butter':  lambda: butter(N, Wn, btype=btype, fs=self.fs, output='sos'),
            'cheby1':  lambda: cheby1(N, rp, Wn, btype=btype, fs=self.fs, output='sos'),
            'cheby2':  lambda: cheby2(N, rs, Wn, btype=btype, fs=self.fs, output='sos'),
            'ellip':   lambda: ellip(N, rp, rs, Wn, btype=btype, fs=self.fs, output='sos'),
            'bessel':  lambda: bessel(N, Wn, btype=btype, fs=self.fs, output='sos'),
        }
        if ftype not in design_funcs:
            raise ValueError(f"Unknown ftype: {ftype}. Choose from {list(design_funcs.keys())}")
        return design_funcs[ftype]()

    def lowpass(self, cutoff_period, N=4, ftype='butter', **kwargs):
        """低通滤波器 — 周期 > cutoff_period 被保留"""
        Wn = self._p2f(cutoff_period)
        return self._design(N, Wn, 'lowpass', ftype, **kwargs)

    def highpass(self, cutoff_period, N=4, ftype='butter', **kwargs):
        """高通滤波器 — 周期 < cutoff_period 被保留"""
        Wn = self._p2f(cutoff_period)
        return self._design(N, Wn, 'highpass', ftype, **kwargs)

    def bandpass(self, period_low, period_high, N=4, ftype='butter', **kwargs):
        """带通滤波器"""
        Wn = [self._p2f(period_high), self._p2f(period_low)]
        return self._design(N, Wn, 'bandpass', ftype, **kwargs)

    def bandstop(self, period_low, period_high, N=4, ftype='butter', **kwargs):
        """带阻滤波器"""
        Wn = [self._p2f(period_high), self._p2f(period_low)]
        return self._design(N, Wn, 'bandstop', ftype, **kwargs)

    @staticmethod
    def apply(sos, x, zero_phase=True):
        """应用滤波器"""
        if zero_phase:
            return sosfiltfilt(sos, x)
        return sosfilt(sos, x)

    def analyze(self, sos, title='IIR Filter'):
        """绘制频率响应+群延迟+零极点分析"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 8))

        # 频率响应
        w, h = sosfreqz(sos, worN=2048, fs=self.fs)
        h_db = 20 * np.log10(np.maximum(np.abs(h), 1e-15))
        axes[0, 0].plot(w, h_db, linewidth=1.2, color='steelblue')
        axes[0, 0].axhline(y=0, color='gray', linewidth=0.5)
        axes[0, 0].axhline(y=-3, color='red', linewidth=0.5, linestyle='--', alpha=0.6)
        axes[0, 0].set_xlabel('频率'); axes[0, 0].set_ylabel('幅度 (dB)')
        axes[0, 0].set_title(f'{title} — 频率响应')
        axes[0, 0].set_ylim(-80, 5)
        axes[0, 0].grid(True, alpha=0.3)

        # 群延迟
        freqs, gd = group_delay(sos, fs=self.fs)
        axes[0, 1].plot(freqs, gd, linewidth=1.2, color='darkorange')
        axes[0, 1].set_xlabel('频率'); axes[0, 1].set_ylabel('群延迟 (样本)')
        axes[0, 1].set_title('群延迟')
        axes[0, 1].grid(True, alpha=0.3)

        # 零极点图
        theta = np.linspace(0, 2*np.pi, 200)
        axes[1, 0].plot(np.cos(theta), np.sin(theta), 'k-', linewidth=0.8, alpha=0.5)
        # 从 SOS 提取零极点
        for section in sos:
            b_s, a_s = section[:3], section[3:]
            # 补零到正确长度
            b_pad = np.concatenate([b_s, [0, 0]]) if len(b_s) < 5 else b_s
            a_pad = np.concatenate([a_s, [0, 0]]) if len(a_s) < 5 else a_s
            z, p, _ = tf2zpk(b_pad, a_pad)
            axes[1, 0].plot(np.real(z), np.imag(z), 'o', markersize=5,
                           markerfacecolor='white', markeredgecolor='steelblue')
            axes[1, 0].plot(np.real(p), np.imag(p), 'x', markersize=8, color='crimson')
        axes[1, 0].set_xlabel('实部'); axes[1, 0].set_ylabel('虚部')
        axes[1, 0].set_title('零极点图')
        axes[1, 0].set_xlim(-1.5, 1.5); axes[1, 0].set_ylim(-1.5, 1.5)
        axes[1, 0].set_aspect('equal')
        axes[1, 0].grid(True, alpha=0.3)

        # 相位响应
        phase = np.unwrap(np.angle(h))
        axes[1, 1].plot(w, phase, linewidth=1.2, color='steelblue')
        axes[1, 1].set_xlabel('频率'); axes[1, 1].set_ylabel('相位 (rad)')
        axes[1, 1].set_title('相位响应 (非线性)')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


# 演示
iid = IIRFilterDesigner(fs=252)
sos_demo = iid.lowpass(cutoff_period=25, N=4, ftype='butter')
y_demo = iid.apply(sos_demo, price)

fig = iid.analyze(sos_demo, title='Butterworth N=4 低通 (周期>25天)')
plt.savefig('code/ch05_fig7_designer_demo.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch05_fig7_designer_demo.png")
print("  IIRFilterDesigner 类已封装完毕。")

# ============================================================
# Part 8: 实时流式滤波演示（状态保持）
# ============================================================
print("\n" + "=" * 60)
print("Part 8: 实时流式滤波 — 逐批处理，保持状态")
print("=" * 60)

# 模拟实时数据流：每次接收 10 个新数据点
np.random.seed(99)
stream_signal = np.sin(2 * np.pi * 5 * np.arange(500) / fs) + 0.3 * np.random.randn(500)

# 初始化滤波器状态
sos_stream = butter(4, 10, btype='low', fs=fs, output='sos')
zi = np.zeros((sos_stream.shape[0], 2))  # 初始状态为零

chunk_size = 20
output_stream = np.zeros(len(stream_signal))

print(f"\n  模拟实时数据流：总长 {len(stream_signal)} 个点，每次处理 {chunk_size} 个点")
print(f"  滤波器状态在批次之间保持（zi 参数）")

for start in range(0, len(stream_signal), chunk_size):
    end = min(start + chunk_size, len(stream_signal))
    chunk = stream_signal[start:end]
    out_chunk, zi = sosfilt(sos_stream, chunk, zi=zi)
    output_stream[start:end] = out_chunk
    if start < 100:
        print(f"    批次 {start//chunk_size}: 处理 [{start}:{end}], 输出形状保持")

# 与一次性滤波对比（应完全相同）
output_batch = sosfilt(sos_stream, stream_signal)
difference = np.max(np.abs(output_stream - output_batch))
print(f"\n  逐批处理 vs 一次性处理的最大差异: {difference:.2e}")
print(f"  → {'✓ 完全等价' if difference < 1e-12 else '✗ 有差异'}")

fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
axes[0].plot(stream_signal[:150], linewidth=0.4, color='gray', alpha=0.6, label='原始流')
axes[0].plot(output_stream[:150], linewidth=1.2, color='steelblue', label='实时滤波输出')
axes[0].set_ylabel('值')
axes[0].set_title('实时流式滤波 — 逐批处理 (Butterworth N=4 lowpass 10Hz)')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].plot(output_stream[:150] - output_batch[:150], linewidth=1, color='crimson')
axes[1].set_xlabel('样本序号')
axes[1].set_ylabel('差异')
axes[1].set_title(f'逐批 vs 一次性滤波的差异 (最大 = {difference:.2e})')
axes[1].grid(True, alpha=0.3)

fig.suptitle('实时流式滤波 — sosfilt + zi 状态保持',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch05_fig8_streaming_filter.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch05_fig8_streaming_filter.png")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("第5章运行完毕！")
print("=" * 60)
print(f"""
📋 本章产出：

  笔记：doc/05_IIR滤波器设计.md
  代码：code/ch05_IIR滤波器设计.py
  图片：
    ch05_fig1_iir_prototypes.png       — 五种 IIR 原型同阶数全面对比
    ch05_fig2_elliptic_sweep.png       — Elliptic rp/rs 参数扫参
    ch05_fig3_butterworth_orders.png   — Butterworth 阶数扫参
    ch05_fig4_pole_zero.png            — 零极点稳定性图
    ch05_fig5_sos_vs_ba.png            — SOS vs (b,a) 数值精度对比
    ch05_fig6_stock_iir_vs_fir.png     — IIR 4阶 vs FIR 61阶 股价趋势提取
    ch05_fig7_designer_demo.png        — IIRFilterDesigner 工具类
    ch05_fig8_streaming_filter.png     — 实时流式滤波演示

🎯 核心收获：
  1. IIR 用反馈回路以极低阶数实现陡峭过渡（4阶 vs FIR 60+阶）
  2. 五种原型 = 五种"平整度 vs 陡峭度"的权衡方式
  3. Butterworth = 默认首选；Elliptic = 最陡峭但不保波形
  4. 高阶 IIR 必须用 SOS 格式（数值精度）— output='sos' + sosfilt
  5. IIR 非线性相位可用 filtfilt 弥补（离线场景）
  6. 零极点图 = IIR 稳定性的可视化判别
  7. sosfilt + zi = 实时流式滤波的正确姿势

📖 下一站：第6章 — 滤波器应用实战
  → lfilter / filtfilt / sosfilt / sosfiltfilt 全面对比
  → 实时 vs 离线，设计→应用的完整闭环
""")
