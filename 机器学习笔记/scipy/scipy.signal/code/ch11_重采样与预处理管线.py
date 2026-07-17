#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第11章 · 重采样与预处理管线 — 配套代码
=============================================
场景驱动：降采样 / 去趋势 / 标准化 / 构建管线

演示1: decimate vs 直接抽取（抗混叠的价值）
演示2: resample vs resample_poly（非整数倍）
演示3: detrend — 去趋势前后频谱对比
演示4: 标准化策略对比
演示5: SignalPreprocessor — 可复用的链式管线

运行方式：
  python code/ch11_重采样与预处理管线.py

依赖：
  pip install numpy scipy matplotlib
"""

import numpy as np
from scipy import signal
from scipy.signal import (decimate, resample, resample_poly, upfirdn,
                          detrend, firwin, filtfilt)
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.dpi': 120, 'font.size': 9,
    'axes.titlesize': 11, 'axes.labelsize': 9,
})
print("=" * 60)
print("第11章 · 重采样与预处理管线")
print("=" * 60)

# ============================================================
# 演示1: decimate vs 直接抽取 — 抗混叠的价值
# ============================================================
print("\n" + "=" * 60)
print("演示1: decimate vs 直接抽取 — 为什么不能直接隔点取")
print("=" * 60)

np.random.seed(42)
fs_orig = 1000
t_orig = np.arange(4000) / fs_orig

# 信号：40Hz（安全） + 80Hz（超过新Nyquist=50Hz → 会混叠！）
sig = (np.sin(2 * np.pi * 40 * t_orig)
       + 0.5 * np.sin(2 * np.pi * 80 * t_orig))

# 降采样到 100Hz（q=10）
q = 10
fs_new = fs_orig / q

# 错误做法
sig_bad = sig[::q]

# 正确做法
sig_good = decimate(sig, q=q, n=40, ftype='fir')

# 频谱对比
fig, axes = plt.subplots(3, 1, figsize=(16, 9))

for ax, (data, fs_val, title, color) in zip(
    axes,
    [(sig, fs_orig,
      f'原始信号 (fs={fs_orig}Hz) — 40Hz + 80Hz',
      'gray'),
     (sig_bad, fs_new,
      f'直接抽取 → 80Hz 混叠成 {abs(fs_new-80):.0f}Hz 假信号！',
      'crimson'),
     (sig_good, fs_new,
      'decimate → 80Hz 先被抗混叠滤波去除，只保留 40Hz',
      'steelblue')]):
    freqs = np.fft.rfftfreq(len(data), d=1/fs_val)
    X = np.abs(np.fft.rfft(data))
    mask = freqs <= fs_val/2 * 0.9
    ax.plot(freqs[mask], X[mask], linewidth=0.8, color=color)
    ax.axvline(x=40, color='green', linewidth=1, linestyle='--', alpha=0.7)
    ax.axvline(x=fs_val/2, color='orange', linewidth=0.8, linestyle=':', alpha=0.7,
               label=f'Nyquist={fs_val/2:.0f}Hz')
    ax.set_ylabel('幅度')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('频率 (Hz)')
fig.suptitle('降采样对比 — 直接抽取产生混叠假峰，decimate 安全',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch11_fig1_decimate.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch11_fig1_decimate.png")

# ============================================================
# 演示2: resample vs resample_poly
# ============================================================
print("\n" + "=" * 60)
print("演示2: resample vs resample_poly — 非整数倍重采样")
print("=" * 60)

fs2 = 1000
t2 = np.arange(2000) / fs2
sig2 = np.sin(2 * np.pi * 10 * t2) + 0.3 * np.sin(2 * np.pi * 45 * t2)

# 目标：256 Hz → 100 Hz（非整数倍 2.56）
target_fs = 100
target_n = int(len(sig2) * target_fs / fs2)

# FFT 方法
sig_fft = resample(sig2, target_n)

# 多相滤波方法
sig_poly = resample_poly(sig2, up=1, down=10, window=('kaiser', 5.0))
# 注意：resample_poly 的 up/down 必须是整数，这里只能近似 1000→100

# 对比时域
fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
p = slice(0, 200)
t_fft = np.arange(len(sig_fft)) / target_fs
t_poly = np.arange(len(sig_poly)) / (fs2/10)

axes[0].plot(t2[p], sig2[p], linewidth=0.8, color='gray', alpha=0.5, label='原始 1000Hz')
axes[0].plot(t_fft[:140], sig_fft[:140], linewidth=0.8, color='steelblue',
             label=f'resample(FFT) → {target_fs}Hz')
axes[0].set_ylabel('幅值')
axes[0].set_title('resample (FFT法) — 快速但假设信号周期性')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].plot(t2[p], sig2[p], linewidth=0.8, color='gray', alpha=0.5, label='原始 1000Hz')
axes[1].plot(t_poly[:min(140, len(sig_poly))],
             sig_poly[:min(140, len(sig_poly))],
             linewidth=0.8, color='darkorange',
             label=f'resample_poly → ~{fs2/10:.0f}Hz')
axes[1].set_xlabel('时间 (s)')
axes[1].set_ylabel('幅值')
axes[1].set_title('resample_poly (多相滤波) — 更精确但只有理数倍率')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

fig.suptitle('重采样方法对比', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch11_fig2_resample.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch11_fig2_resample.png")

# ============================================================
# 演示3: detrend — 去趋势前后频谱对比
# ============================================================
print("\n" + "=" * 60)
print("演示3: detrend — 去掉趋势让频谱更清晰")
print("=" * 60)

np.random.seed(88)
N3 = 1000
t3 = np.arange(N3)

# 信号：线性趋势 + 周期成分 + 噪声
trend = 0.05 * t3
periodic = 3 * np.sin(2 * np.pi * t3 / 50)
noise = 0.5 * np.random.randn(N3)
sig3 = trend + periodic + noise

# 去趋势
sig3_detrended = detrend(sig3, type='linear')

# 频谱对比
fig, axes = plt.subplots(2, 2, figsize=(16, 8))

# 时域 — 去趋势前
axes[0, 0].plot(t3, sig3, linewidth=0.5, color='steelblue')
axes[0, 0].plot(t3, trend, linewidth=1, color='crimson', linestyle='--',
                alpha=0.7, label='线性趋势')
axes[0, 0].set_xlabel('样本序号'); axes[0, 0].set_ylabel('值')
axes[0, 0].set_title('原始信号 — 趋势 + 周期 + 噪声')
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True, alpha=0.3)

# 频谱 — 去趋势前
freqs = np.fft.rfftfreq(N3)
X_raw = np.abs(np.fft.rfft(sig3))
axes[0, 1].plot(freqs[1:80], X_raw[1:80], linewidth=0.8, color='crimson')
axes[0, 1].set_xlabel('归一化频率'); axes[0, 1].set_ylabel('幅度')
axes[0, 1].set_title('原始频谱 — 低频趋势能量掩盖了周期峰')
axes[0, 1].grid(True, alpha=0.3)

# 时域 — 去趋势后
axes[1, 0].plot(t3, sig3_detrended, linewidth=0.5, color='steelblue')
axes[1, 0].axhline(y=0, color='gray', linewidth=0.5)
axes[1, 0].set_xlabel('样本序号'); axes[1, 0].set_ylabel('值')
axes[1, 0].set_title('去线性趋势后 — 只剩下周期 + 噪声')
axes[1, 0].grid(True, alpha=0.3)

# 频谱 — 去趋势后
X_det = np.abs(np.fft.rfft(sig3_detrended))
axes[1, 1].plot(freqs[1:80], X_det[1:80], linewidth=0.8, color='steelblue')
axes[1, 1].axvline(x=1/50, color='green', linewidth=1, linestyle='--', alpha=0.7,
                   label='周期=50 的频率')
axes[1, 1].set_xlabel('归一化频率'); axes[1, 1].set_ylabel('幅度')
axes[1, 1].set_title('去趋势后频谱 — 周期成分的峰清晰可见')
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

fig.suptitle('去趋势的效果 — 低频能量不再淹没周期成分',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch11_fig3_detrend.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch11_fig3_detrend.png")

# ============================================================
# 演示4: 标准化策略对比
# ============================================================
print("\n" + "=" * 60)
print("演示4: 标准化 — 让不同量纲的信号可比较")
print("=" * 60)

np.random.seed(55)
t4 = np.arange(500)
# 温度信号：范围 [-5, 35]
temp = 15 + 20 * np.sin(2 * np.pi * t4 / 200) + 2 * np.random.randn(500)
# 湿度信号：范围 [30, 90]
humid = 60 + 30 * np.sin(2 * np.pi * t4 / 200 + 1) + 3 * np.random.randn(500)

# 三种标准化
temp_z = (temp - np.mean(temp)) / np.std(temp)
humid_z = (humid - np.mean(humid)) / np.std(humid)

temp_mm = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))
humid_mm = (humid - np.min(humid)) / (np.max(humid) - np.min(humid))

temp_rms = temp / np.sqrt(np.mean(temp**2))
humid_rms = humid / np.sqrt(np.mean(humid**2))

fig, axes = plt.subplots(2, 2, figsize=(16, 8))

axes[0, 0].plot(t4, temp, linewidth=0.8, label=f'温度 (均值={np.mean(temp):.1f})')
axes[0, 0].plot(t4, humid, linewidth=0.8, label=f'湿度 (均值={np.mean(humid):.1f})')
axes[0, 0].set_title('原始 — 量纲不同，无法直接比较')
axes[0, 0].legend(fontsize=8); axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(t4, temp_z, linewidth=0.8, label='温度 (Z-score)')
axes[0, 1].plot(t4, humid_z, linewidth=0.8, label='湿度 (Z-score)')
axes[0, 1].set_title('Z-score — 均值为0，标准差为1')
axes[0, 1].legend(fontsize=8); axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(t4, temp_mm, linewidth=0.8, label='温度 (Min-max)')
axes[1, 0].plot(t4, humid_mm, linewidth=0.8, label='湿度 (Min-max)')
axes[1, 0].set_title('Min-max — 缩放到 [0, 1]')
axes[1, 0].legend(fontsize=8); axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(t4, temp_rms, linewidth=0.8, label='温度 (RMS)')
axes[1, 1].plot(t4, humid_rms, linewidth=0.8, label='湿度 (RMS)')
axes[1, 1].set_title('RMS归一化 — 能量归一')
axes[1, 1].legend(fontsize=8); axes[1, 1].grid(True, alpha=0.3)

fig.suptitle('标准化策略对比 — 选择取决于你的分析目的',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch11_fig4_normalize.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch11_fig4_normalize.png")

# ============================================================
# 演示5: SignalPreprocessor — 完整管线
# ============================================================
print("\n" + "=" * 60)
print("演示5: SignalPreprocessor — 链式预处理管线")
print("=" * 60)


class SignalPreprocessor:
    """可复用的信号预处理管线，支持链式调用。"""

    def __init__(self, fs):
        self.fs = fs
        self.steps = []
        self._log = []

    def detrend(self, dtype='linear'):
        self.steps.append(('detrend', dtype))
        return self

    def lowpass(self, cutoff_period, numtaps=51):
        self.steps.append(('lowpass', cutoff_period, numtaps))
        return self

    def decimate(self, q):
        self.steps.append(('decimate', q))
        return self

    def normalize(self, method='zscore'):
        self.steps.append(('normalize', method))
        return self

    def process(self, x):
        self._log = []
        fs_current = self.fs
        for step in self.steps:
            name = step[0]
            if name == 'detrend':
                x = detrend(x, type=step[1])
                self._log.append(f'detrend(type={step[1]}) → shape={x.shape}')

            elif name == 'lowpass':
                cutoff_freq = fs_current / step[1]
                b = firwin(step[2], cutoff_freq, fs=fs_current)
                x = filtfilt(b, [1], x)
                self._log.append(
                    f'lowpass(cutoff={cutoff_freq:.1f}Hz, n={step[2]}) → shape={x.shape}')

            elif name == 'decimate':
                x = decimate(x, q=step[1], ftype='fir')
                fs_current /= step[1]
                self._log.append(
                    f'decimate(q={step[1]}) → shape={x.shape}, fs→{fs_current:.0f}Hz')

            elif name == 'normalize':
                if step[1] == 'zscore':
                    x = (x - np.mean(x)) / np.std(x)
                elif step[1] == 'minmax':
                    x = (x - np.min(x)) / (np.max(x) - np.min(x))
                self._log.append(f'normalize(method={step[1]}) → shape={x.shape}')

        return x

    def show_log(self):
        for entry in self._log:
            print(f"    {entry}")


# 构造测试数据：15分钟采样的电力负荷（96点/天）
np.random.seed(100)
fs_load = 96
N_load = 96 * 14  # 14天
t_load = np.arange(N_load) / fs_load

# 趋势 + 日周期 + 半日周期 + 噪声
trend_l = 0.01 * t_load
daily = 0.5 * np.sin(2 * np.pi * t_load / 24)
half_day = 0.15 * np.sin(2 * np.pi * t_load / 12)
noise_l = 0.08 * np.random.randn(N_load)
raw = trend_l + daily + half_day + noise_l

# 构建管线并处理
pp = SignalPreprocessor(fs=fs_load)
pp.detrend('linear') \
  .lowpass(cutoff_period=8, numtaps=41) \
  .decimate(q=4) \
  .normalize('zscore')

processed = pp.process(raw)

print(f"\n  原始数据: {len(raw)} 点 @ {fs_load} 点/天")
print(f"  处理后:   {len(processed)} 点 @ {fs_load/4:.0f} 点/天")
print(f"  处理步骤:")
pp.show_log()

# 图：前后对比
fig, axes = plt.subplots(2, 1, figsize=(16, 7))

p_raw = slice(0, 4 * 96)  # 前4天
axes[0].plot(t_load[p_raw], raw[p_raw], linewidth=0.5, color='steelblue')
axes[0].set_xlabel('时间 (天)')
axes[0].set_ylabel('值')
axes[0].set_title(f'原始数据 — 14天电力负荷 (fs={fs_load}点/天)')
axes[0].grid(True, alpha=0.3)

t_processed = np.arange(len(processed)) / (fs_load / 4)
axes[1].plot(t_processed, processed, linewidth=0.8, color='crimson')
axes[1].set_xlabel('时间 (天)')
axes[1].set_ylabel('值 (Z-score)')
axes[1].set_title(f'预处理后 — 去趋势+低通+降采样4倍+Z-score (fs={fs_load/4:.0f}点/天)')
axes[1].grid(True, alpha=0.3)

fig.suptitle('SignalPreprocessor 管线 — 4步处理，链式调用',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch11_fig5_pipeline.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch11_fig5_pipeline.png")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("第11章运行完毕！")
print("=" * 60)
print(f"""
📋 本章产出：

  笔记：doc/11_重采样与预处理管线.md
  代码：code/ch11_重采样与预处理管线.py
  图片：
    ch11_fig1_decimate.png     — 直接抽取 vs decimate（抗混叠）
    ch11_fig2_resample.png     — resample vs resample_poly
    ch11_fig3_detrend.png      — 去趋势前后频谱对比
    ch11_fig4_normalize.png    — 三种标准化策略对比
    ch11_fig5_pipeline.png     — SignalPreprocessor 完整管线

🎯 核心收获：
  1. decimate = 降采样最省心选择（自动抗混叠）
  2. resample_poly = 非整数倍重采样
  3. detrend = FFT 前必做（去 DC/线性趋势）
  4. Z-score = 最通用的标准化
  5. 管线顺序：去趋势 → 滤波 → 降采样 → 归一化
  6. 降采样后记住更新 fs

📖 下一站：第12章 — 信号去噪方法对比
  → Savitzky-Golay / 中值滤波 / Wiener / 小波阈值
""")
