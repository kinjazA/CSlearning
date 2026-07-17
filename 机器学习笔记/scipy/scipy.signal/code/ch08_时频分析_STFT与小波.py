#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第8章 · 时频分析（STFT, 频谱图, 小波）— 配套代码
=====================================================
场景驱动，每个场景演示一种时频分析工具的使用。

场景1: chirp 信号的时频轨迹 — STFT + 频谱图
场景2: 时间 vs 频率分辨率的权衡 — 不同窗长的对比
场景3: 多尺度分析 — CWT (小波变换) vs STFT
场景4: 瞬时频率追踪 — 从频谱图提取频率轨迹

运行方式：
  python code/ch08_时频分析_STFT与小波.py

依赖：
  pip install numpy scipy matplotlib
"""

import numpy as np
from scipy import signal
from scipy.signal import stft, istft, spectrogram, cwt, ricker, morlet2
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.dpi': 120, 'font.size': 9,
    'axes.titlesize': 11, 'axes.labelsize': 9,
})
print("=" * 60)
print("第8章 · 时频分析（STFT, 频谱图, 小波）")
print("=" * 60)

# ============================================================
# 场景1: chirp — 用频谱图追踪频率随时间的变化
# ============================================================
print("\n" + "=" * 60)
print("场景1: chirp 信号 — 时频轨迹一目了然")
print("=" * 60)

fs = 1000
duration = 5.0
t = np.arange(0, duration, 1/fs)

# 构造信号：线性 chirp (10Hz→200Hz) + 一个瞬时 300Hz 的短脉冲
chirp_sig = signal.chirp(t, f0=10, t1=duration, f1=200, method='linear')
# 在 t=3s 处插入一个短促的 300Hz 脉冲
pulse_pos = int(3.0 * fs)
chirp_sig[pulse_pos:pulse_pos+50] += 0.5 * np.sin(2 * np.pi * 300 * t[:50])

print(f"  信号: 线性 chirp 10→200Hz + t=3s处一个300Hz脉冲")
print(f"  FFT 能看出频谱，但看不出'频率随时间的变化'")

# FFT（全时段）— 看不到时间信息
X_full = np.abs(np.fft.rfft(chirp_sig))
freqs_full = np.fft.rfftfreq(len(chirp_sig), d=1/fs)

# STFT — 时频谱
nperseg = 256
f_stft, t_stft, Zxx = stft(chirp_sig, fs=fs, window='hann',
                            nperseg=nperseg, noverlap=200)
spec = np.abs(Zxx)

# 图
fig, axes = plt.subplots(3, 1, figsize=(16, 9))

# 时域（前 1 秒 + 全貌缩略）
axes[0].plot(t[:600], chirp_sig[:600], linewidth=0.4, color='steelblue')
axes[0].set_xlabel('时间 (s)')
axes[0].set_ylabel('幅值')
axes[0].set_title('时域波形（前 0.6s）— 看到频率在变，但看不清"怎么变"')
axes[0].grid(True, alpha=0.3)

# FFT — 全时段频谱
axes[1].plot(freqs_full[freqs_full <= 350],
             X_full[freqs_full <= 350], linewidth=0.8, color='steelblue')
axes[1].set_xlabel('频率 (Hz)')
axes[1].set_ylabel('幅度')
axes[1].set_title('FFT 全时段频谱 — 能看出 10-200Hz 有能量，但不知道何时出现')
axes[1].grid(True, alpha=0.3)

# 频谱图
im = axes[2].pcolormesh(t_stft, f_stft, 20 * np.log10(np.maximum(spec, 1e-10)),
                        shading='gouraud', cmap='inferno')
axes[2].set_xlabel('时间 (s)')
axes[2].set_ylabel('频率 (Hz)')
axes[2].set_title('频谱图 — x=时间, y=频率, 颜色=强度。chirp 的扫频轨迹清晰可见！')
axes[2].set_ylim(0, 400)
plt.colorbar(im, ax=axes[2], label='幅度 (dB)')

fig.suptitle('场景1 — 频谱图揭示 chirp 的频率如何随时间扫动',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch08_fig1_chirp_spectrogram.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch08_fig1_chirp_spectrogram.png")

# ============================================================
# 场景2: 时间 vs 频率分辨率 — 窗长的权衡
# ============================================================
print("\n" + "=" * 60)
print("场景2: 窗长选择 — 时间分辨率 vs 频率分辨率")
print("=" * 60)

# 构造测试信号：两个非常接近的频率在交替出现
t_test = np.arange(0, 2.0, 1/fs)
# 前1秒：50Hz
sig1 = np.sin(2 * np.pi * 50 * t_test) * (t_test < 1.0)
# 后1秒：55Hz（非常接近50Hz）
sig2 = np.sin(2 * np.pi * 55 * t_test) * (t_test >= 1.0)
test_signal = sig1 + sig2 + 0.05 * np.random.randn(len(t_test))

# 三种窗长
windows = [
    (64, 48, '短窗 (64点) — 时间精度高'),
    (256, 192, '中窗 (256点) — 折中'),
    (1024, 768, '长窗 (1024点) — 频率精度高'),
]

fig, axes = plt.subplots(3, 1, figsize=(16, 10))

for ax, (nseg, novlp, desc) in zip(axes, windows):
    f, t_s, Sxx = spectrogram(test_signal, fs=fs, window='hann',
                               nperseg=nseg, noverlap=novlp)
    im = ax.pcolormesh(t_s, f, 20 * np.log10(np.maximum(Sxx, 1e-10)),
                       shading='gouraud', cmap='inferno')
    ax.axvline(x=1.0, color='cyan', linewidth=1.5, linestyle='--', alpha=0.8,
               label='频率切换点 (50→55Hz)')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('频率 (Hz)')
    ax.set_title(desc)
    ax.set_ylim(30, 80)
    ax.legend(fontsize=8)
    plt.colorbar(im, ax=ax, label='dB')

fig.suptitle('场景2 — 窗长越长，频率越清晰但时间越模糊（注意切换点的扩散程度）',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch08_fig2_window_tradeoff.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch08_fig2_window_tradeoff.png")
print("""
  解读：
    短窗(左上)：切换点精确可见，但 50Hz 和 55Hz 几乎区分不开
    中窗(中)：  折中——能大致区分两个频率，切换时间也还好
    长窗(下)：  50Hz 和 55Hz 分得很清楚，但切换点被模糊成一个宽带
""")

# ============================================================
# 场景3: CWT 小波变换 — 可变尺度的时频分析
# ============================================================
print("\n" + "=" * 60)
print("场景3: CWT 小波变换 — 低频精频率，高频精时间")
print("=" * 60)

# 构造一个"多尺度"信号
np.random.seed(123)
t_cwt = np.arange(0, 4.0, 1/fs)
N_cwt = len(t_cwt)

# 低频成分（约 5 Hz，持续整个时长）
sig_low = np.sin(2 * np.pi * 5 * t_cwt)
# 高频瞬态（150 Hz，只出现在 t=1.5s 和 t=3.0s 附近）
sig_high = np.zeros(N_cwt)
p1 = int(1.5 * fs)
p2 = int(3.0 * fs)
sig_high[p1:p1+40] = 0.6 * np.sin(2 * np.pi * 150 * t_cwt[:40])
sig_high[p2:p2+40] = 0.4 * np.sin(2 * np.pi * 150 * t_cwt[:40])
# 噪声
noise_cwt = 0.1 * np.random.randn(N_cwt)

cwt_signal = sig_low + sig_high + noise_cwt

# STFT（用于对比）
f_stft2, t_stft2, Sxx_stft2 = spectrogram(cwt_signal, fs=fs,
                                           window='hann',
                                           nperseg=512, noverlap=384)

# CWT
widths = np.arange(1, 128)
cwt_result = cwt(cwt_signal, ricker, widths)
# 尺度→频率的近似映射
freqs_cwt = fs / (2 * np.pi * widths)  # 近似频率

print(f"  信号: 5Hz 持续低频 + 150Hz 短暂瞬态脉冲")
print(f"  STFT 用固定窗长 → 无法同时解析好低频和高频")
print(f"  CWT 用可变尺度 → 低频时宽窗(精频率)，高频时窄窗(精时间)")

# 图
fig, axes = plt.subplots(3, 1, figsize=(16, 10))

# 时域
axes[0].plot(t_cwt[:2000], cwt_signal[:2000], linewidth=0.4, color='steelblue')
axes[0].axvline(x=1.5, color='red', linewidth=0.8, linestyle='--', alpha=0.7)
axes[0].axvline(x=3.0, color='red', linewidth=0.8, linestyle='--', alpha=0.7)
axes[0].set_xlabel('时间 (s)')
axes[0].set_ylabel('幅值')
axes[0].set_title('时域 — 低频背景 + 两个瞬态脉冲（红虚线处）')
axes[0].grid(True, alpha=0.3)

# STFT 频谱图
im1 = axes[1].pcolormesh(t_stft2, f_stft2,
                         20 * np.log10(np.maximum(Sxx_stft2, 1e-10)),
                         shading='gouraud', cmap='inferno')
axes[1].set_xlabel('时间 (s)')
axes[1].set_ylabel('频率 (Hz)')
axes[1].set_title('STFT 频谱图 — 低频的 5Hz 模糊成一片，脉冲位置也扩散了')
axes[1].set_ylim(0, 200)
plt.colorbar(im1, ax=axes[1], label='dB')

# CWT
im2 = axes[2].pcolormesh(t_cwt, freqs_cwt, np.abs(cwt_result),
                         shading='gouraud', cmap='inferno')
axes[2].set_xlabel('时间 (s)')
axes[2].set_ylabel('近似频率 (Hz)')
axes[2].set_title('CWT (ricker) — 低频 5Hz 分辨率好，高频脉冲定位精确')
axes[2].set_ylim(0, 200)
plt.colorbar(im2, ax=axes[2], label='幅度')

fig.suptitle('场景3 — CWT 小波变换：同时获得低频和高频的最优分辨率',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch08_fig3_cwt_vs_stft.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch08_fig3_cwt_vs_stft.png")

# ============================================================
# 场景4: 瞬时频率追踪
# ============================================================
print("\n" + "=" * 60)
print("场景4: 瞬时频率追踪 — 从频谱图提取频率轨迹")
print("=" * 60)

# 构造线性 chirp：5→50 Hz
chirp_tracking = signal.chirp(t, f0=5, t1=duration, f1=50, method='linear')
chirp_tracking += 0.1 * np.random.randn(len(t))

# 计算频谱图
f_tr, t_tr, Sxx_tr = spectrogram(chirp_tracking, fs=fs, window='hann',
                                  nperseg=256, noverlap=200)

# 简单瞬时频率追踪：每个时刻取能量最大的频率
peak_idx = np.argmax(Sxx_tr, axis=0)
f_instant = f_tr[peak_idx]

# 真实频率线（已知，因为是我们构造的）
f_true = 5 + (50 - 5) * t_tr / duration

# 图
fig, axes = plt.subplots(2, 1, figsize=(16, 8))

# 频谱图 + 追踪线
im = axes[0].pcolormesh(t_tr, f_tr, 20 * np.log10(np.maximum(Sxx_tr, 1e-10)),
                        shading='gouraud', cmap='inferno')
axes[0].plot(t_tr, f_true, 'w--', linewidth=2, alpha=0.8, label='真实频率')
axes[0].plot(t_tr, f_instant, 'c-', linewidth=1.5, alpha=0.8, label='追踪频率')
axes[0].set_xlabel('时间 (s)')
axes[0].set_ylabel('频率 (Hz)')
axes[0].set_title('频谱图 + 瞬时频率追踪')
axes[0].set_ylim(0, 70)
axes[0].legend(fontsize=9)
plt.colorbar(im, ax=axes[0], label='dB')

# 追踪误差
axes[1].plot(t_tr, f_instant - f_true, linewidth=0.8, color='crimson')
axes[1].axhline(y=0, color='gray', linewidth=0.5)
axes[1].set_xlabel('时间 (s)')
axes[1].set_ylabel('频率误差 (Hz)')
axes[1].set_title('瞬时频率追踪误差 — 大部分时刻在 ±2 Hz 以内')
axes[1].grid(True, alpha=0.3)

fig.suptitle('场景4 — 从频谱图中提取瞬时频率轨迹',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch08_fig4_instantaneous_freq.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch08_fig4_instantaneous_freq.png")

# ============================================================
# 附加演示：istft — 从时频谱恢复信号
# ============================================================
print("\n" + "=" * 60)
print("附加: istft — 从时频谱恢复信号（验证无信息丢失）")
print("=" * 60)

_, _, Zxx_roundtrip = stft(chirp_sig[:fs*2], fs=fs, window='hann',
                            nperseg=256, noverlap=200)
_, sig_roundtrip = istft(Zxx_roundtrip, fs=fs, window='hann',
                          nperseg=256, noverlap=200)

# 截断到相同长度比较
min_len = min(len(chirp_sig[:fs*2]), len(sig_roundtrip))
reconstruction_error = np.max(np.abs(
    chirp_sig[:min_len] - sig_roundtrip[:min_len]
))
print(f"  STFT → ISTFT 重建误差: {reconstruction_error:.2e}")
print(f"  → {'✓ 几乎无损' if reconstruction_error < 1e-10 else '注意：重叠区域外的点有损失'}")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("第8章运行完毕！")
print("=" * 60)
print(f"""
📋 本章产出：

  笔记：doc/08_时频分析_STFT与小波.md（场景驱动）
  代码：code/ch08_时频分析_STFT与小波.py（场景驱动）
  图片：
    ch08_fig1_chirp_spectrogram.png       — chirp 的时频轨迹
    ch08_fig2_window_tradeoff.png         — 窗长 vs 时间/频率分辨率
    ch08_fig3_cwt_vs_stft.png             — CWT 小波 vs STFT
    ch08_fig4_instantaneous_freq.png      — 频谱图 + 瞬时频率追踪

🎯 核心收获：
  1. FFT 告诉你"有什么频率"但不告诉你"频率何时出现"
  2. STFT/频谱图 = 滑动窗 FFT → x=时间, y=频率, 颜色=强度
  3. 窗长 = 时间精度 vs 频率精度的权衡（物理定律，无法两全）
  4. CWT = 可变尺度 —— 低频精频率，高频精时间
  5. ricker 小波适合检测瞬态，morlet2 适合检测振荡
  6. 瞬时频率追踪 = 频谱图中每列找最大能量的频率

📖 下一站：第9章 — 峰值检测与信号特征提取
  → 从信号中自动提取有意义的"事件"（峰、谷、拐点）
  → 搭建从信号到 ML 特征矩阵的桥梁
""")
