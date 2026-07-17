#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第9章 · 峰值检测与信号特征提取 — 配套代码
=================================================
场景驱动，逐个参数演示 find_peaks 的用法。

演示1: prominence — 最重要的参数（天然区分真峰和噪声）
演示2: height / distance / width — 辅助筛选
演示3: 全参数综合调参 — 从宽松到收紧
演示4: find_peaks_cwt — 高噪声下的鲁棒检测
演示5: 包络提取 — 连峰成线

运行方式：
  python code/ch09_峰值检测与信号特征提取.py

依赖：
  pip install numpy scipy matplotlib
"""

import numpy as np
from scipy import signal
from scipy.signal import (find_peaks, find_peaks_cwt,
                          peak_prominences, peak_widths,
                          argrelmax, argrelmin)
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.dpi': 120, 'font.size': 9,
    'axes.titlesize': 11, 'axes.labelsize': 9,
})
print("=" * 60)
print("第9章 · 峰值检测与信号特征提取")
print("=" * 60)

# ============================================================
# 演示1: prominence — 理解"突出度"
# ============================================================
print("\n" + "=" * 60)
print("演示1: prominence — 峰有多'突出'？")
print("=" * 60)

# 构造信号：一个大峰 + 几个小凸起
np.random.seed(42)
x = np.linspace(0, 10, 500)
# 主峰
main_peak = 5 * np.exp(-((x - 5) ** 2) / 0.5)
# 两个小凸起（prominence 不同）
small_bump1 = 1.2 * np.exp(-((x - 2) ** 2) / 0.1)  # 小但孤立
small_bump2 = 0.8 * np.exp(-((x - 6.5) ** 2) / 0.05)  # 更小且紧挨主峰
noise = 0.15 * np.random.randn(len(x))

sig = main_peak + small_bump1 + small_bump2 + noise

# 检测所有的峰（不加任何筛选）
all_peaks, _ = find_peaks(sig)
# 只用 prominence 筛选
peaks_prom, props_prom = find_peaks(sig, prominence=0.5)

# 计算每个候选峰的 prominence
prom_all = peak_prominences(sig, all_peaks)[0]

print(f"  不设参数 → 检测到 {len(all_peaks)} 个峰（包含噪声凸起）")
print(f"  prominence≥0.5 → 检测到 {len(peaks_prom)} 个峰（只保留真正的峰）")
print(f"\n  各候选峰的 prominence：")
for i, (p, prom) in enumerate(zip(all_peaks, prom_all)):
    marker = " ✓" if p in peaks_prom else ""
    print(f"    位置{x[p]:.2f}: prominence={prom:.2f}{marker}")

# 图
fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

# 上图：所有峰（不加筛选）
axes[0].plot(x, sig, linewidth=0.8, color='steelblue')
axes[0].plot(x[all_peaks], sig[all_peaks], 'o', markersize=6,
             color='crimson', alpha=0.7, label=f'{len(all_peaks)} peaks (未筛选)')
axes[0].set_ylabel('幅值')
axes[0].set_title('不加筛选 — 噪声的微凸起也被检测为峰')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# 下图：只用 prominence 筛选
axes[1].plot(x, sig, linewidth=0.8, color='steelblue')
axes[1].plot(x[peaks_prom], sig[peaks_prom], 'o', markersize=10,
             color='green', alpha=0.8, label=f'{len(peaks_prom)} peaks (prominence≥0.5)')
# 标注每个被保留峰的 prominence
for p in peaks_prom:
    prom = props_prom['prominences'][list(peaks_prom).index(p)]
    axes[1].annotate(f'prom={prom:.1f}',
                     (x[p], sig[p]),
                     textcoords="offset points", xytext=(0, 15),
                     ha='center', fontsize=8, color='green')
# 标注被排除的峰
excluded = set(all_peaks) - set(peaks_prom)
for p in excluded:
    axes[1].annotate(f'被排除\nprom={prom_all[list(all_peaks).index(p)]:.1f}',
                     (x[p], sig[p]),
                     textcoords="offset points", xytext=(0, -25),
                     ha='center', fontsize=7, color='red')
axes[1].set_xlabel('x')
axes[1].set_ylabel('幅值')
axes[1].set_title('prominence≥0.5 — 自动排除了噪声凸起（突出不够的峰）')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

fig.suptitle('prominence — 区分"真正的峰"和"噪声的随机凸起"的核心参数',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch09_fig1_prominence.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch09_fig1_prominence.png")

# ============================================================
# 演示2: height / distance / width — 辅助筛选
# ============================================================
print("\n" + "=" * 60)
print("演示2: height / distance / width — 精细化控制")
print("=" * 60)

# 构造信号：多个不同特征的峰
np.random.seed(55)
t = np.linspace(0, 20, 1000)
sig2 = (np.sin(2 * np.pi * t / 5)             # 主周期
        + 0.3 * np.sin(2 * np.pi * t / 0.8)   # 短周期（会被 distance 过滤）
        + 0.5 * np.exp(-((t - 10) ** 2) / 2)  # 宽包络
        + 0.15 * np.random.randn(len(t)))

# 四组参数对比
configs = [
    ({}, '不设参数'),
    ({'height': 0.5}, 'height≥0.5'),
    ({'distance': 30}, 'distance≥30'),
    ({'width': 5}, 'width≥5'),
]

fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True, sharey=True)

for ax, (params, desc) in zip(axes, configs):
    peaks, props = find_peaks(sig2, **params)
    ax.plot(t, sig2, linewidth=0.6, color='steelblue')
    ax.plot(t[peaks], sig2[peaks], 'o', markersize=5,
            color='crimson', alpha=0.8)
    ax.set_ylabel('幅值')
    ax.set_title(f'{desc} → {len(peaks)} 个峰')
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('t')
fig.suptitle('各参数独立演示 — 不同维度筛选出不同类型的峰',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch09_fig2_parameters.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch09_fig2_parameters.png")

# ============================================================
# 演示3: 综合调参 — 从宽松到收紧
# ============================================================
print("\n" + "=" * 60)
print("演示3: 综合调参 — 逐步收紧约束")
print("=" * 60)

# 构造类 ECG 信号
np.random.seed(33)
t_ecg = np.linspace(0, 10, 1000)
# 模拟心跳：每80个采样点一个QRS波
ecg = np.zeros(1000)
for i in range(0, 1000, 80):
    if i + 40 < 1000:
        ecg[i+20:i+35] += 1.5 * np.sin(np.linspace(0, np.pi, 15))
ecg += 0.05 * np.random.randn(1000)

# 三步调参
# Step1: 不设参数
p1, _ = find_peaks(ecg)
# Step2: prominence
p2, _ = find_peaks(ecg, prominence=0.3)
# Step3: prominence + distance（最终）
p3, props3 = find_peaks(ecg, prominence=0.3, distance=50, width=3)

print(f"  Step1 不设参数:    {len(p1)} 个峰")
print(f"  Step2 +prominence:  {len(p2)} 个峰")
print(f"  Step3 +distance:    {len(p3)} 个峰")

# 图
fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)

for ax, (peaks, _, desc) in zip(axes,
    [(p1, None, f'Step1 未筛选 → {len(p1)} peaks（大量噪声误检）'),
     (p2, None, f'Step2 prominence≥0.3 → {len(p2)} peaks（噪声减少）'),
     (p3, props3, f'Step3 +distance≥50 +width≥3 → {len(p3)} peaks（只有真正的R波）')]):
    ax.plot(t_ecg, ecg, linewidth=0.5, color='steelblue')
    ax.plot(t_ecg[peaks], ecg[peaks], 'o', markersize=6,
            color='crimson', alpha=0.8)
    ax.set_ylabel('幅值')
    ax.set_title(desc)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('时间 (任意单位)')
fig.suptitle('综合调参 — 从一大堆误检到精确检测 R 波',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch09_fig3_tuning.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch09_fig3_tuning.png")

# ============================================================
# 演示4: find_peaks_cwt — 高噪声下的鲁棒检测
# ============================================================
print("\n" + "=" * 60)
print("演示4: find_peaks_cwt — 噪声很大时")
print("=" * 60)

# 构造高噪声信号
np.random.seed(77)
t_noisy = np.linspace(0, 15, 800)
true_peaks_positions = np.array([100, 250, 400, 600])
clean_signal = np.zeros(800)
for pos in true_peaks_positions:
    clean_signal[pos:pos+20] = 2 * np.sin(np.linspace(0, np.pi, 20))
noisy_signal = clean_signal + 0.6 * np.random.randn(800)

# 标准 find_peaks（调 prominence）
peaks_std, _ = find_peaks(noisy_signal, prominence=0.5)

# CWT 方法
peaks_cwt = find_peaks_cwt(noisy_signal, widths=np.arange(5, 30))

print(f"  真实峰位置: {true_peaks_positions}")
print(f"  find_peaks (prominence=0.5):  {peaks_std}")
print(f"  find_peaks_cwt:               {np.array(peaks_cwt)}")

# 图
fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

axes[0].plot(noisy_signal, linewidth=0.5, color='gray', alpha=0.6, label='含噪信号')
axes[0].plot(clean_signal, linewidth=0.8, color='black', linestyle='--', alpha=0.5,
             label='真实峰位置')
axes[0].plot(peaks_std, noisy_signal[peaks_std], 'o', markersize=6,
             color='crimson', alpha=0.8, label=f'find_peaks ({len(peaks_std)}个)')
for tp in true_peaks_positions:
    axes[0].axvline(x=tp, color='green', linewidth=0.8, linestyle=':', alpha=0.6)
axes[0].set_ylabel('幅值')
axes[0].set_title(f'find_peaks (prominence=0.5) — 检测到 {len(peaks_std)} 个峰')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].plot(noisy_signal, linewidth=0.5, color='gray', alpha=0.6, label='含噪信号')
axes[1].plot(clean_signal, linewidth=0.8, color='black', linestyle='--', alpha=0.5,
             label='真实峰位置')
axes[1].plot(peaks_cwt, noisy_signal[peaks_cwt], 'o', markersize=6,
             color='steelblue', alpha=0.8, label=f'find_peaks_cwt ({len(peaks_cwt)}个)')
for tp in true_peaks_positions:
    axes[1].axvline(x=tp, color='green', linewidth=0.8, linestyle=':', alpha=0.6)
axes[1].set_xlabel('采样点')
axes[1].set_ylabel('幅值')
axes[1].set_title(f'find_peaks_cwt — 检测到 {len(peaks_cwt)} 个峰')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

fig.suptitle('高噪声场景 — find_peaks_cwt 更鲁棒（绿虚线=真实峰位置）',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch09_fig4_cwt.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch09_fig4_cwt.png")

# ============================================================
# 演示5: 包络提取 + argrelmax
# ============================================================
print("\n" + "=" * 60)
print("演示5: 包络提取 — 连峰成线")
print("=" * 60)

# 构造一个幅度调制的信号
t_env = np.linspace(0, 20, 2000)
carrier = np.sin(2 * np.pi * t_env * 2)  # 2 Hz 载波
amplitude_mod = 1 + 0.6 * np.sin(2 * np.pi * t_env * 0.1)  # 0.1 Hz 幅度调制
modulated = amplitude_mod * carrier + 0.05 * np.random.randn(len(t_env))

# 方法1：找上峰 → 插值连成上包络
peaks_up, _ = find_peaks(modulated, distance=15)
upper_envelope = np.interp(t_env, t_env[peaks_up], modulated[peaks_up])

# 方法2：用 argrelmin 找谷（下包络）
valleys = argrelmin(modulated, order=20)[0]
lower_envelope = np.interp(t_env, t_env[valleys], modulated[valleys])

# 图
fig, axes = plt.subplots(2, 1, figsize=(16, 8))

p = slice(0, 600)
axes[0].plot(t_env[p], modulated[p], linewidth=0.4, color='steelblue', label='信号')
axes[0].plot(t_env[p], upper_envelope[p], linewidth=1.5, color='crimson',
             label='上包络')
axes[0].plot(t_env[p], lower_envelope[p], linewidth=1.5, color='darkorange',
             label='下包络')
axes[0].plot(t_env[p], amplitude_mod[p], linewidth=0.6, color='black',
             linestyle='--', alpha=0.6, label='真实包络 (已知)')
axes[0].set_ylabel('幅值')
axes[0].set_title('幅度调制信号 + 包络提取')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# 上包络的峰检测
peaks_envelope, props_env = find_peaks(upper_envelope, prominence=0.1)
axes[1].plot(t_env, upper_envelope, linewidth=0.8, color='crimson', label='上包络')
axes[1].plot(t_env[peaks_envelope], upper_envelope[peaks_envelope], 'o',
             markersize=8, color='steelblue', label=f'包络的峰 ({len(peaks_envelope)}个)')
axes[1].set_xlabel('t')
axes[1].set_ylabel('幅值')
axes[1].set_title('对上包络再做峰检测 — 发现幅度调制的周期模式')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

fig.suptitle('包络提取 — 先在信号上找峰，再连峰成线',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch09_fig5_envelope.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch09_fig5_envelope.png")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("第9章运行完毕！")
print("=" * 60)
print(f"""
📋 本章产出：

  笔记：doc/09_峰值检测与信号特征提取.md（场景驱动）
  代码：code/ch09_峰值检测与信号特征提取.py（场景驱动）
  图片：
    ch09_fig1_prominence.png    — prominence：区分真峰和噪声
    ch09_fig2_parameters.png    — height/distance/width 独立演示
    ch09_fig3_tuning.png        — ECG 综合调参：从宽松到收紧
    ch09_fig4_cwt.png           — find_peaks_cwt 高噪声鲁棒检测
    ch09_fig5_envelope.png      — 包络提取 + 峰检测

🎯 核心收获：
  1. prominence = 峰的最重要属性 → 天然区分真峰和噪声凸起
  2. 调参顺序：先 prominence → 再加 distance → 最后 height/width
  3. find_peaks_cwt = 高噪声时的鲁棒替代
  4. 包络 = 连峰成线 → 提取信号的"轮廓"

📖 下一站：第10章 — LTI系统分析与Hilbert变换
  → 用数学模型描述物理系统，用Hilbert变换提取瞬时频率和精确包络
""")
