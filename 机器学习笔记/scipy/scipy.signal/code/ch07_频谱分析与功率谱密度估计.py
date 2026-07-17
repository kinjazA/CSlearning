#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第7章 · 频谱分析与功率谱密度估计 — 配套代码
=================================================
按场景组织，每个场景演示一个核心频谱分析技术。

场景1: 隐藏周期检测 — FFT 幅度谱
场景2: 稳健谱估计 — Welch 方法 vs 周期图
场景3: 非均匀采样 — Lomb-Scargle 周期图
场景4: 双信号频率相关性 — 相干性 (Coherence) & CSD
场景5: 频谱泄漏与加窗 — 为什么 FFT 前要加窗

运行方式：
  python code/ch07_频谱分析与功率谱密度估计.py

依赖：
  pip install numpy scipy matplotlib
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, rfft, rfftfreq, fftfreq
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.dpi': 120, 'font.size': 9,
    'axes.titlesize': 11, 'axes.labelsize': 9,
})
print("=" * 60)
print("第7章 · 频谱分析与功率谱密度估计")
print("=" * 60)

# ============================================================
# 场景1: 隐藏周期检测 — FFT 幅度谱
# ============================================================
print("\n" + "=" * 60)
print("场景1: 隐藏周期检测 — FFT 让你'看见'频率")
print("=" * 60)

# 构造信号：三个频率成分 + 噪声
np.random.seed(42)
fs = 500
duration = 4.0
N = int(fs * duration)
t = np.arange(N) / fs

# 成分1：50 Hz 强信号
sig_50hz = 3.0 * np.sin(2 * np.pi * 50 * t)
# 成分2：120 Hz 中等信号
sig_120hz = 1.5 * np.sin(2 * np.pi * 120 * t)
# 成分3：180 Hz 弱信号（故意让它很难从时域看到）
sig_180hz = 0.4 * np.sin(2 * np.pi * 180 * t)
# 噪声
noise = 1.0 * np.random.randn(N)

composite = sig_50hz + sig_120hz + sig_180hz + noise

# FFT
X = rfft(composite)
freqs = rfftfreq(N, d=1/fs)
amplitude = np.abs(X) / N
amplitude[1:-1] *= 2  # 补偿单边谱

print(f"  信号包含: 50Hz(强) + 120Hz(中等) + 180Hz(弱) + 噪声")
print(f"  采样率={fs}Hz, 时长={duration}s, N={N}")
print(f"  频率分辨率 = fs/N = {fs/N:.2f} Hz")

# 图：时域 + 频域
fig, axes = plt.subplots(2, 1, figsize=(16, 7))

# 时域
axes[0].plot(t[:300], composite[:300], linewidth=0.5, color='steelblue')
axes[0].set_xlabel('时间 (s)')
axes[0].set_ylabel('幅值')
axes[0].set_title('时域波形（前300点）— 能看出有几个频率成分吗？很难！')
axes[0].grid(True, alpha=0.3)

# 频域
axes[1].plot(freqs, amplitude, linewidth=0.8, color='crimson')
axes[1].set_xlabel('频率 (Hz)')
axes[1].set_ylabel('幅度')
axes[1].set_title('FFT 幅度谱 — 50Hz, 120Hz, 180Hz 一目了然！')
axes[1].set_xlim(0, 250)
# 标注三个峰值
for f, label in [(50, '50Hz'), (120, '120Hz'), (180, '180Hz (弱)')]:
    axes[1].axvline(x=f, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    axes[1].text(f, amplitude[int(f * N / fs)] + 0.05, label,
                 fontsize=8, ha='center', color='red')
axes[1].grid(True, alpha=0.3)

fig.suptitle('场景1 — 时域看不清的周期，FFT 让它们一目了然',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch07_fig1_fft_spectrum.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch07_fig1_fft_spectrum.png")

# ============================================================
# 场景2: Welch vs 周期图 — 为什么不能用原始 FFT？
# ============================================================
print("\n" + "=" * 60)
print("场景2: Welch 方法 — 分段平均降低方差")
print("=" * 60)

# 构造一个随机信号（粉红噪声：低频能量高，高频能量低）
# 粉红噪声的 PSD ∝ 1/f，对数坐标下是一条斜线
np.random.seed(99)
N_psd = 8192
pink_noise = np.cumsum(np.random.randn(N_psd))  # 累积求和 ≈ 粉红噪声
pink_noise = pink_noise / np.std(pink_noise)

# 周期图（单次 FFT + 平方）
f_per, psd_per = signal.periodogram(pink_noise, fs=1000)

# Welch 方法 — 不同 nperseg
welch_configs = [
    (128, 64, '小段(128) — 方差小，分辨率低'),
    (512, 256, '中段(512) — 折中'),
    (2048, 1024, '大段(2048) — 方差大，分辨率高'),
]
welch_results = []
for nperseg, noverlap, desc in welch_configs:
    f_w, psd_w = signal.welch(pink_noise, fs=1000,
                              nperseg=nperseg, noverlap=noverlap,
                              window='hann')
    welch_results.append((f_w, psd_w, desc))

# 图：周期图 + 三种 Welch
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 周期图
axes[0, 0].semilogy(f_per, psd_per, linewidth=0.3, color='crimson')
axes[0, 0].set_xlabel('频率 (Hz)')
axes[0, 0].set_ylabel('PSD (V²/Hz)')
axes[0, 0].set_title('周期图 — 单次 FFT，方差极大（非常"吵"）')
axes[0, 0].grid(True, alpha=0.3)

# 三种 Welch
for ax, (f_w, psd_w, desc) in zip([axes[0, 1], axes[1, 0], axes[1, 1]],
                                   welch_results):
    ax.semilogy(f_w, psd_w, linewidth=0.8, color='steelblue')
    ax.set_xlabel('频率 (Hz)')
    ax.set_ylabel('PSD (V²/Hz)')
    ax.set_title(desc)
    ax.grid(True, alpha=0.3)

fig.suptitle('场景2 — 周期图 vs Welch 方法：分段平均让谱估计光滑可靠',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch07_fig2_welch_vs_periodogram.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch07_fig2_welch_vs_periodogram.png")
print("""
  解读：
    周期图（左上）：毛刺极多，几乎看不出规律
    Welch 小段（右上）：非常光滑，但每个"峰"都很宽——分辨率低
    Welch 中段（左下）：折中——可见大致趋势，不太吵
    Welch 大段（右下）：逼近周期图——峰窄但毛刺变多

  → 选择 nperseg 就是选择"分辨率"和"方差"的权衡点。
""")

# ============================================================
# 场景3: Lomb-Scargle — 非均匀采样
# ============================================================
print("\n" + "=" * 60)
print("场景3: Lomb-Scargle — 数据不等间隔也能做频谱")
print("=" * 60)

# 模拟非均匀采样场景
# 比如：只在"观测窗口"内才有数据（天文观测类比）
np.random.seed(66)
N_total = 500
t_uniform = np.linspace(0, 100, N_total)

# 真实信号：周期约 10（频率 ≈ 0.1）
true_signal = np.sin(2 * np.pi * t_uniform / 10)

# 模拟非均匀采样：随机选取 60% 的观测点
n_obs = int(N_total * 0.6)
obs_indices = np.sort(np.random.choice(N_total, n_obs, replace=False))
t_obs = t_uniform[obs_indices]
x_obs = true_signal[obs_indices] + 0.3 * np.random.randn(n_obs)

# 两个方法对比：
# 方法A（错误）：在缺失位置填 NaN，然后用等间隔方法的思维去处理
# 方法B（正确）：Lomb-Scargle

# 要考察的频率范围
freqs_ls = np.linspace(0.02, 0.5, 300)
pgram_ls = signal.lombscargle(t_obs, x_obs, freqs_ls, normalize=True)

print(f"  总采样点: {N_total}, 实际观测: {n_obs} ({n_obs/N_total*100:.0f}%)")
print(f"  真实周期约 10（频率 ≈ 0.1）")

# 图
fig, axes = plt.subplots(3, 1, figsize=(16, 9))

# 均匀采样的完整信号（参考）
axes[0].plot(t_uniform, true_signal, linewidth=0.6, color='steelblue')
axes[0].set_ylabel('幅值')
axes[0].set_title('真实完整信号（通常我们看不到这个）— 周期≈10')
axes[0].grid(True, alpha=0.3)

# 实际观测（非均匀）
axes[1].plot(t_obs, x_obs, 'o', markersize=3, color='darkorange', alpha=0.7)
axes[1].set_ylabel('幅值')
axes[1].set_title(f'实际观测 — 只有 {n_obs} 个非均匀采样点')
axes[1].grid(True, alpha=0.3)

# Lomb-Scargle 结果
axes[2].plot(freqs_ls, pgram_ls, linewidth=1.0, color='crimson')
axes[2].axvline(x=0.1, color='black', linewidth=1, linestyle='--',
                alpha=0.7, label='真实频率 0.1 (周期=10)')
axes[2].set_xlabel('频率')
axes[2].set_ylabel('Lomb-Scargle 功率')
axes[2].set_title('Lomb-Scargle 周期图 — 峰值准确地出现在频率 0.1 处')
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3)

fig.suptitle('场景3 — 非均匀采样也能做频谱：Lomb-Scargle 周期图',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch07_fig3_lombscargle.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch07_fig3_lombscargle.png")

# ============================================================
# 场景4: 相干性 — 两个信号在哪些频率上同步？
# ============================================================
print("\n" + "=" * 60)
print("场景4: 相干性 — 两信号在不同频带的"相关系数"")
print("=" * 60)

# 构造两个信号
np.random.seed(123)
N_coh = 4096
fs_coh = 500
t_coh = np.arange(N_coh) / fs_coh

# 公共低频成分 — 两信号共享（高度相干）
common_low = np.sin(2 * np.pi * 10 * t_coh)

# 信号A：低频共享 + 中频独有 + 噪声
sig_a = (common_low
         + 0.5 * np.sin(2 * np.pi * 50 * t_coh)   # A 独有的 50 Hz
         + 0.3 * np.random.randn(N_coh))

# 信号B：低频共享 + 不同的中频 + 噪声
sig_b = (common_low
         + 0.5 * np.sin(2 * np.pi * 80 * t_coh)   # B 独有的 80 Hz
         + 0.3 * np.random.randn(N_coh))

# 计算相干性
f_coh, coh = signal.coherence(sig_a, sig_b, fs=fs_coh,
                               nperseg=512, noverlap=256)

# 计算 CSD
f_csd, csd_val = signal.csd(sig_a, sig_b, fs=fs_coh,
                             nperseg=512, noverlap=256)

# 全局相关系数（时域）
global_corr = np.corrcoef(sig_a, sig_b)[0, 1]

print(f"  两信号的全局相关系数: {global_corr:.3f}")
print(f"  但它们在低频 10Hz 处高度相干，在中频（50/80Hz）处互不相关")

# 图
fig, axes = plt.subplots(3, 1, figsize=(16, 9))

# 时域（前 500 点）
plot_n = 500
axes[0].plot(t_coh[:plot_n], sig_a[:plot_n], linewidth=0.5, color='steelblue',
             alpha=0.8, label='信号 A')
axes[0].plot(t_coh[:plot_n], sig_b[:plot_n], linewidth=0.5, color='darkorange',
             alpha=0.8, label='信号 B')
axes[0].set_ylabel('幅值')
axes[0].set_title(f'时域波形 — 全局相关系数 = {global_corr:.3f}')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# 各自的 PSD
f_a, psd_a = signal.welch(sig_a, fs=fs_coh, nperseg=512)
f_b, psd_b = signal.welch(sig_b, fs=fs_coh, nperseg=512)
axes[1].semilogy(f_a, psd_a, linewidth=0.8, color='steelblue', label='信号 A PSD')
axes[1].semilogy(f_b, psd_b, linewidth=0.8, color='darkorange', label='信号 B PSD')
axes[1].axvline(x=10, color='green', linewidth=1, linestyle='--', alpha=0.7,
                label='10Hz (共享)')
axes[1].axvline(x=50, color='steelblue', linewidth=0.5, linestyle=':', alpha=0.5,
                label='50Hz (A独有)')
axes[1].axvline(x=80, color='darkorange', linewidth=0.5, linestyle=':', alpha=0.5,
                label='80Hz (B独有)')
axes[1].set_ylabel('PSD')
axes[1].set_title('各自的功率谱密度 — 10Hz 是共有的峰')
axes[1].legend(fontsize=7)
axes[1].grid(True, alpha=0.3)

# 相干性
axes[2].plot(f_coh, coh, linewidth=1.2, color='crimson')
axes[2].axhline(y=0.5, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
axes[2].axvline(x=10, color='green', linewidth=1, linestyle='--', alpha=0.7,
                label='10Hz — 高相干(>0.9)')
axes[2].set_xlabel('频率 (Hz)')
axes[2].set_ylabel('相干性')
axes[2].set_title('相干性 = 分频率的相关系数 — 只在 10Hz 附近高度同步')
axes[2].set_ylim(0, 1.05)
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

fig.suptitle('场景4 — 相干性：总相关系数掩盖了不同频带的差异',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch07_fig4_coherence.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch07_fig4_coherence.png")

# ============================================================
# 场景5: 频谱泄漏与加窗
# ============================================================
print("\n" + "=" * 60)
print("场景5: 为什么 FFT 前要加窗 — 频谱泄漏演示")
print("=" * 60)

# 用纯正弦波演示：如果信号不是整数个周期 → 泄漏
fs_leak = 1000
N_leak = 500

# 情况A：整数个周期 → 无泄漏
# 50 Hz, 500个点@1000Hz → 500/1000×50 = 25个完整周期 ✓
t_leak_ok = np.arange(N_leak) / fs_leak
f_ok = 50
sine_ok = np.sin(2 * np.pi * f_ok * t_leak_ok)

# 情况B：非整数个周期 → 泄漏！
# 50.5 Hz, 500个点@1000Hz → 500/1000×50.5 = 25.25个周期 ✗
f_bad = 50.5
sine_bad = np.sin(2 * np.pi * f_bad * t_leak_ok)

# 不加窗 vs 加 Hann 窗
X_ok_raw = np.abs(rfft(sine_ok))
X_bad_raw = np.abs(rfft(sine_bad))
X_bad_win = np.abs(rfft(sine_bad * signal.windows.hann(N_leak)))

freqs_leak = rfftfreq(N_leak, d=1/fs_leak)

print(f"  情况A：{f_ok}Hz, 恰好 {N_leak/fs_leak*f_ok:.0f} 个完整周期 → 无泄漏")
print(f"  情况B：{f_bad}Hz, 只有 {N_leak/fs_leak*f_bad:.1f} 个周期 → 端点不匹配 → 泄漏")

# 图
fig, axes = plt.subplots(2, 2, figsize=(16, 8))

# 时域：整数周期（端点连续）
axes[0, 0].plot(t_leak_ok[:100], sine_ok[:100], linewidth=0.8, color='steelblue')
axes[0, 0].plot([0, t_leak_ok[N_leak-1]], [sine_ok[0], sine_ok[-1]],
                'ro', markersize=6, label='首尾相接 ✓')
axes[0, 0].set_xlabel('时间 (s)')
axes[0, 0].set_title(f'情况A: {f_ok}Hz — 整数个周期，首尾平滑相接')
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True, alpha=0.3)

# 频域：无泄漏
axes[0, 1].plot(freqs_leak[:100], X_ok_raw[:100], linewidth=0.8, color='steelblue')
axes[0, 1].axvline(x=f_ok, color='green', linewidth=1, linestyle='--', label=f'{f_ok}Hz')
axes[0, 1].set_xlabel('频率 (Hz)')
axes[0, 1].set_title('FFT — 能量集中在 50Hz，无泄漏')
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(True, alpha=0.3)

# 时域：非整数周期（端点不连续）
axes[1, 0].plot(t_leak_ok[:100], sine_bad[:100], linewidth=0.8, color='crimson')
axes[1, 0].plot([0, t_leak_ok[N_leak-1]], [sine_bad[0], sine_bad[-1]],
                'ro', markersize=6, label='首尾不连续 ✗')
axes[1, 0].set_xlabel('时间 (s)')
axes[1, 0].set_title(f'情况B: {f_bad}Hz — 非整数个周期，首尾有跳变')
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

# 频域：泄漏 vs 加窗
axes[1, 1].plot(freqs_leak[:100], X_bad_raw[:100], linewidth=0.8, color='crimson',
                alpha=0.6, label='不加窗 — 泄漏严重')
axes[1, 1].plot(freqs_leak[:100], X_bad_win[:100], linewidth=0.8, color='steelblue',
                label='加 Hann 窗 — 泄漏大幅减少')
axes[1, 1].axvline(x=f_bad, color='green', linewidth=1, linestyle='--', label=f'{f_bad}Hz')
axes[1, 1].set_xlabel('频率 (Hz)')
axes[1, 1].set_title('FFT — 不加窗能量泄漏到邻频，加窗后大幅改善')
axes[1, 1].legend(fontsize=7)
axes[1, 1].grid(True, alpha=0.3)

fig.suptitle('场景5 — 频谱泄漏：为什么 FFT 前要加窗',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch07_fig5_spectral_leakage.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch07_fig5_spectral_leakage.png")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("第7章运行完毕！")
print("=" * 60)
print(f"""
📋 本章产出：

  笔记：doc/07_频谱分析与功率谱密度估计.md（场景驱动）
  代码：code/ch07_频谱分析与功率谱密度估计.py（场景驱动）
  图片：
    ch07_fig1_fft_spectrum.png           — 时域隐藏的周期，FFT 一目了然
    ch07_fig2_welch_vs_periodogram.png   — 周期图 vs Welch（分段平均）
    ch07_fig3_lombscargle.png            — 非均匀采样的频谱分析
    ch07_fig4_coherence.png              — 两信号在不同频带的同步程度
    ch07_fig5_spectral_leakage.png       — 为什么 FFT 前要加窗

🎯 核心收获：
  1. FFT = 信号从"时间表示"切换到"频率表示"
  2. 幅度谱 = 每个频率的强度 → 方差大、不稳定
  3. Welch PSD = 分段平均 → 方差小、稳定 → 工业标准
  4. nperseg 控制"分辨率 vs 方差"的权衡
  5. Lomb-Scargle = 非均匀采样数据的频谱工具
  6. 相干性 = 分频率的"相关系数" → 比全局相关更细致
  7. FFT 前加窗 → 减少频谱泄漏

📖 下一站：第8章 — 时频分析（STFT, 频谱图, 小波）
  → 当频率成分随时间变化时，你需要时间+频率的二维视图
""")
