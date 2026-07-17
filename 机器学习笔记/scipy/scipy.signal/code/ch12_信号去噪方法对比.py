#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第12章 · 信号去噪方法对比 — 配套代码
=========================================
场景驱动，同一条含噪信号用五种方法去噪，横向对比。

演示1: Savitzky-Golay — 保边缘的平滑
演示2: 中值滤波 — 去除脉冲尖峰
演示3: Wiener — 自适应去噪
演示4: 五种方法横向对比 — 同信号不同噪声
演示5: 去噪效果量化评估

运行方式：
  python code/ch12_信号去噪方法对比.py

依赖：
  pip install numpy scipy matplotlib
"""

import numpy as np
from scipy import signal
from scipy.signal import (savgol_filter, medfilt, wiener,
                          firwin, filtfilt)
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.dpi': 120, 'font.size': 9,
    'axes.titlesize': 11, 'axes.labelsize': 9,
})
print("=" * 60)
print("第12章 · 信号去噪方法对比")
print("=" * 60)

# ============================================================
# 演示1: Savitzky-Golay — 阶数 vs 窗口长度
# ============================================================
print("\n" + "=" * 60)
print("演示1: Savitzky-Golay — 保边缘的平滑")
print("=" * 60)

np.random.seed(42)
N1 = 300
x1 = np.arange(N1)

# 构造"干净信号"：平坦区 + 几个尖峰
clean = np.zeros(N1)
clean[50:70] = 3 * np.exp(-((np.arange(20)-10)**2)/10)   # 窄高斯峰
clean[140:150] = 2.0                                     # 方波跳变
clean[200:240] = np.sin(np.linspace(0, np.pi, 40))       # 半正弦
noisy = clean + 0.4 * np.random.randn(N1)

# 不同参数的 S-G 滤波
configs = [
    (savgol_filter(noisy, window_length=11, polyorder=3),
     'window=11, order=3 (较平滑)'),
    (savgol_filter(noisy, window_length=11, polyorder=5),
     'window=11, order=5 (更追踪尖峰)'),
    (savgol_filter(noisy, window_length=21, polyorder=3),
     'window=21, order=3 (更宽平滑)'),
]

fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)

for ax, (y, desc) in zip(axes, configs):
    ax.plot(x1, noisy, linewidth=0.2, color='gray', alpha=0.4, label='含噪')
    ax.plot(x1, clean, linewidth=0.6, color='black', linestyle='--',
            alpha=0.6, label='真实信号')
    ax.plot(x1, y, linewidth=1.0, color='steelblue', label=desc)
    mse = np.mean((y - clean)**2)
    ax.set_ylabel('值')
    ax.set_title(f'{desc} — MSE={mse:.4f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('样本序号')

# 标注关键区域
for ax in axes:
    ax.axvspan(50, 70, alpha=0.06, color='green')
    ax.axvspan(140, 150, alpha=0.06, color='orange')
axes[0].text(60, 2.8, '窄峰', fontsize=8, ha='center', color='green')
axes[0].text(145, 2.2, '跳变', fontsize=8, ha='center', color='orange')

fig.suptitle('Savitzky-Golay — window 和 polyorder 对保边缘能力的影响',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch12_fig1_savgol.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch12_fig1_savgol.png")

# ============================================================
# 演示2: 中值滤波 — 脉冲噪声克星
# ============================================================
print("\n" + "=" * 60)
print("演示2: 中值滤波 — 专治脉冲尖峰")
print("=" * 60)

np.random.seed(66)
N2 = 300
x2 = np.arange(N2)

# 基础信号 + 高斯噪声
base = np.sin(2 * np.pi * x2 / 60)
noisy2 = base + 0.2 * np.random.randn(N2)

# 随机插入尖峰脉冲
spike_positions = np.random.choice(N2, size=8, replace=False)
noisy2_spike = noisy2.copy()
noisy2_spike[spike_positions] += np.random.choice([-3, 3], size=8)

# 三种去噪
y_med3 = medfilt(noisy2_spike, kernel_size=3)
y_med7 = medfilt(noisy2_spike, kernel_size=7)
y_lowpass = filtfilt(firwin(31, 0.15), [1], noisy2_spike)  # 低通对比

fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)

axes[0].plot(x2, base, linewidth=0.6, color='black', linestyle='--', alpha=0.6,
             label='真实信号')
axes[0].plot(x2, noisy2_spike, linewidth=0.5, color='crimson', alpha=0.7,
             label=f'含脉冲噪声 ({len(spike_positions)}个尖峰)')
axes[0].set_ylabel('值')
axes[0].set_title('原始信号 — 高斯噪声 + 偶发尖峰脉冲')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].plot(x2, base, linewidth=0.6, color='black', linestyle='--', alpha=0.6,
             label='真实信号')
axes[1].plot(x2, y_med3, linewidth=1.0, color='steelblue',
             label=f'中值滤波 k=3 (MSE={np.mean((y_med3-base)**2):.4f})')
axes[1].set_ylabel('值')
axes[1].set_title('中值滤波 k=3 — 尖峰基本去除，细节保留好')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

axes[2].plot(x2, base, linewidth=0.6, color='black', linestyle='--', alpha=0.6,
             label='真实信号')
axes[2].plot(x2, y_lowpass, linewidth=1.0, color='darkorange',
             label=f'FIR 低通 (MSE={np.mean((y_lowpass-base)**2):.4f})')
# 标记尖峰残留
for sp in spike_positions:
    axes[2].axvline(x=sp, color='red', linewidth=0.5, linestyle=':', alpha=0.4)
axes[2].set_xlabel('样本序号')
axes[2].set_ylabel('值')
axes[2].set_title('FIR 低通 — 尖峰被"摊平"到邻域，留下涟漪（红虚线=原尖峰位置）')
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

fig.suptitle('中值滤波 vs 低通滤波 — 脉冲噪声的正确应对方式',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch12_fig2_medfilt.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch12_fig2_medfilt.png")

# ============================================================
# 演示3: Wiener — 自适应去噪
# ============================================================
print("\n" + "=" * 60)
print("演示3: Wiener — 平坦区强去噪，变化区弱去噪")
print("=" * 60)

np.random.seed(33)
N3 = 400
x3 = np.arange(N3)

# 信号：平坦区 + 突变区
clean3 = np.zeros(N3)
clean3[80:120] = 2.0                                    # 平顶脉冲（突变边沿）
clean3[180:220] = 2 * np.exp(-((np.arange(40)-20)**2)/50)  # 高斯峰
clean3[280:320] = np.linspace(0, 1.5, 40)               # 线性爬升
noisy3 = clean3 + 0.5 * np.random.randn(N3)

# 去噪
y_sg = savgol_filter(noisy3, window_length=15, polyorder=3)
y_wiener = wiener(noisy3, mysize=11)
b_fir = firwin(31, 0.15)
y_fir = filtfilt(b_fir, [1], noisy3)

# 图
fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)

axes[0].plot(x3, clean3, linewidth=0.8, color='black', linestyle='--', alpha=0.6,
             label='真实信号')
axes[0].plot(x3, noisy3, linewidth=0.3, color='gray', alpha=0.5, label='含噪')
axes[0].set_ylabel('值')
axes[0].set_title(f'原始含噪信号 (噪声 std=0.5)')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

for ax, y, name, color in [
    (axes[1], y_sg, 'Savitzky-Golay (w=15,p=3)', 'steelblue'),
    (axes[2], y_wiener, 'Wiener (size=11)', 'darkorange'),
    (axes[3], y_fir, 'FIR 低通 (n=31)', 'crimson')]:
    ax.plot(x3, clean3, linewidth=0.6, color='black', linestyle='--', alpha=0.5,
            label='真实信号')
    ax.plot(x3, y, linewidth=1.0, color=color, label=name)
    mse = np.mean((y - clean3)**2)
    ax.set_ylabel('值')
    ax.set_title(f'{name} — MSE={mse:.4f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('样本序号')

# 标注信号区域
for ax in axes:
    ax.axvspan(80, 120, alpha=0.05, color='green')
    ax.axvspan(180, 220, alpha=0.05, color='orange')

fig.suptitle('三种去噪对比 — 注意平坦区和突变区的不同表现',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch12_fig3_wiener.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch12_fig3_wiener.png")

# ============================================================
# 演示4: 五种方法横向PK + 量化评估
# ============================================================
print("\n" + "=" * 60)
print("演示4: 五种去噪方法横向PK + 量化评估")
print("=" * 60)

# 复用演示1的信号
methods = {
    '不处理 (基准)':  lambda s: s,
    'FIR 低通 (n=31)': lambda s: filtfilt(firwin(31, 0.15), [1], s),
    'Savitzky-Golay':  lambda s: savgol_filter(s, 11, 3),
    '中值滤波 (k=5)':  lambda s: medfilt(s, 5),
    'Wiener (size=11)': lambda s: wiener(s, mysize=11),
}

results = {}
print(f"\n  {'方法':<22s} {'MSE':>8s} {'相关系数':>8s}")
print(f"  {'-'*40}")
for name, func in methods.items():
    y = func(noisy)
    mse = np.mean((y - clean)**2)
    corr = np.corrcoef(y, clean)[0, 1]
    results[name] = (y, mse, corr)
    print(f"  {name:<22s} {mse:>8.4f} {corr:>8.4f}")

# 图：五种方法波形对比
fig, axes = plt.subplots(5, 1, figsize=(16, 12), sharex=True)

for ax, (name, (y, mse, corr)) in zip(axes, results.items()):
    ax.plot(x1, noisy, linewidth=0.15, color='gray', alpha=0.3)
    ax.plot(x1, clean, linewidth=0.6, color='black', linestyle='--', alpha=0.5)
    color = 'crimson' if name == '不处理 (基准)' else 'steelblue'
    ax.plot(x1, y, linewidth=0.8, color=color)
    ax.set_ylabel('值')
    ax.set_title(f'{name} — MSE={mse:.4f}, r={corr:.4f}')
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('样本序号')
fig.suptitle('五种去噪方法横向对比 — 同信号、不同策略',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch12_fig4_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch12_fig4_comparison.png")

# ============================================================
# 演示5: 残差分析 — 没有 ground truth 时如何评估
# ============================================================
print("\n" + "=" * 60)
print("演示5: 残差分析 — 没真值时怎么判断去噪质量")
print("=" * 60)

# 取一种去噪结果，分析其残差
y_denoised = results['Savitzky-Golay'][0]
residual = noisy - y_denoised

fig, axes = plt.subplots(2, 2, figsize=(16, 8))

# 残差时域
axes[0, 0].plot(x1, residual, linewidth=0.5, color='steelblue')
axes[0, 0].axhline(y=0, color='gray', linewidth=0.5)
axes[0, 0].set_xlabel('样本序号'); axes[0, 0].set_ylabel('残差')
axes[0, 0].set_title('残差时域 — 理想情况应无结构、无趋势')
axes[0, 0].grid(True, alpha=0.3)

# 残差直方图
axes[0, 1].hist(residual, bins=40, density=True, color='steelblue', alpha=0.7,
                edgecolor='white')
# 叠加正态参考
x_pdf = np.linspace(-1.5, 1.5, 200)
axes[0, 1].plot(x_pdf, 1/(0.4*np.sqrt(2*np.pi))*np.exp(-x_pdf**2/(2*0.4**2)),
                'r-', linewidth=1.5, label='正态分布参考')
axes[0, 1].set_xlabel('残差值'); axes[0, 1].set_ylabel('密度')
axes[0, 1].set_title('残差分布 — 接近正态=噪声被有效提取')
axes[0, 1].legend(fontsize=8)

# 残差频谱
freqs_r = np.fft.rfftfreq(len(residual))
X_r = np.abs(np.fft.rfft(residual))
axes[1, 0].plot(freqs_r[1:], X_r[1:], linewidth=0.6, color='steelblue')
axes[1, 0].set_xlabel('归一化频率'); axes[1, 0].set_ylabel('幅度')
axes[1, 0].set_title('残差频谱 — 应无显著峰（噪声不应有周期结构）')
axes[1, 0].grid(True, alpha=0.3)

# 残差 ACF
lags = np.arange(50)
acf_vals = [np.corrcoef(residual[:-l], residual[l:])[0, 1]
            if l > 0 else 1.0 for l in lags]
axes[1, 1].stem(lags, acf_vals, linefmt='steelblue-', markerfmt='steelblueo',
                basefmt='gray')
axes[1, 1].axhline(y=0, color='gray', linewidth=0.5)
axes[1, 1].axhline(y=1.96/np.sqrt(len(residual)), color='red', linewidth=0.8,
                   linestyle='--', alpha=0.7, label='95% 置信边界')
axes[1, 1].axhline(y=-1.96/np.sqrt(len(residual)), color='red', linewidth=0.8,
                   linestyle='--', alpha=0.7)
axes[1, 1].set_xlabel('滞后'); axes[1, 1].set_ylabel('ACF')
axes[1, 1].set_title('残差自相关 — lag>0应接近0（白噪声特征）')
axes[1, 1].legend(fontsize=7)

fig.suptitle('残差分析 — 用四个维度判断去噪质量（无需 ground truth）',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch12_fig5_residual_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch12_fig5_residual_analysis.png")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("第12章运行完毕！")
print("=" * 60)
print(f"""
📋 本章产出：

  笔记：doc/12_信号去噪方法对比.md
  代码：code/ch12_信号去噪方法对比.py
  图片：
    ch12_fig1_savgol.png            — S-G 阶数和窗口长度对保边缘的影响
    ch12_fig2_medfilt.png           — 中值滤波 vs 低通（脉冲噪声对比）
    ch12_fig3_wiener.png            — Wiener 自适应去噪
    ch12_fig4_comparison.png        — 五种方法横向PK + MSE/r量化
    ch12_fig5_residual_analysis.png — 无 ground truth 时的残差四维评估

🎯 核心收获：
  1. Savitzky-Golay → 保边缘平滑（多项式追踪弯曲）
  2. 中值滤波 → 脉冲/尖峰噪声的首选（中位数免疫极端值）
  3. Wiener → 自适应（平坦区强去噪，突变区弱去噪）
  4. 低通 FIR → 持续高斯噪声（精确频率控制）
  5. 残差四维评估：时域结构/分布/频谱/自相关

📖 下一站：第13章 — 信号特征工程与ML集成（最后一章！）
  → 搭建从原始信号到预测模型的完整链路
""")
