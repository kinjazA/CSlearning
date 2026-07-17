#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第13章 · 信号特征工程与ML集成 — 配套代码
=============================================
完整的端到端案例：从原始信号 → 特征提取 → 分类模型

演示：
  1. 构造多类时序数据（正常 + 三种异常模式）
  2. 滑动窗口特征提取（时域 + 频域 + 峰特征）
  3. 特征标准化 + 训练/测试划分
  4. Random Forest 分类 + 特征重要性
  5. 混淆矩阵评估

运行方式：
  python code/ch13_信号特征工程与ML集成.py

依赖：
  pip install numpy scipy matplotlib scikit-learn
"""

import numpy as np
from scipy import signal
from scipy.signal import (welch, find_peaks, savgol_filter,
                          detrend, hilbert)
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay)

plt.rcParams.update({
    'figure.dpi': 120, 'font.size': 9,
    'axes.titlesize': 11, 'axes.labelsize': 9,
})
print("=" * 60)
print("第13章 · 信号特征工程与ML集成（最后一章）")
print("=" * 60)

# ============================================================
# Step 1: 构造多类时序数据
# ============================================================
print("\n" + "=" * 60)
print("Step 1: 构造数据 — 正常 + 三种异常模式")
print("=" * 60)

np.random.seed(42)
fs = 100
duration = 10.0
N_per_signal = int(fs * duration)
n_per_class = 60
classes = ['normal', 'anomaly_spike', 'anomaly_drift', 'anomaly_noise']
n_classes = len(classes)
n_total = n_per_class * n_classes

signals = np.zeros((n_total, N_per_signal))
labels = np.zeros(n_total, dtype=int)
t = np.arange(N_per_signal) / fs

for i in range(n_total):
    label_idx = i // n_per_class
    labels[i] = label_idx
    seed = 42 + i
    np.random.seed(seed)

    # 共同成分：1Hz 基波 + 5Hz 谐波
    base = (np.sin(2 * np.pi * 1 * t)
            + 0.3 * np.sin(2 * np.pi * 5 * t))

    if label_idx == 0:  # 正常
        sig = base + 0.15 * np.random.randn(N_per_signal)

    elif label_idx == 1:  # 异常A：偶发尖峰
        sig = base + 0.15 * np.random.randn(N_per_signal)
        # 随机插入 2-5 个尖峰
        n_spikes = np.random.randint(2, 6)
        for _ in range(n_spikes):
            pos = np.random.randint(100, N_per_signal - 100)
            sig[pos:pos+8] += np.random.uniform(1.0, 2.5)

    elif label_idx == 2:  # 异常B：频率漂移
        drift_rate = np.random.uniform(0.3, 0.8)
        freq_mod = 1 + drift_rate * t
        sig = np.sin(2 * np.pi * freq_mod * t) + 0.15 * np.random.randn(N_per_signal)

    elif label_idx == 3:  # 异常C：高噪声
        sig = base + np.random.uniform(0.4, 0.7) * np.random.randn(N_per_signal)

    signals[i] = sig

print(f"  生成 {n_total} 条信号, 每条 {N_per_signal} 点 @ {fs}Hz")
print(f"  类别: normal({n_per_class}) / spike({n_per_class}) "
      f"/ drift({n_per_class}) / noise({n_per_class})")

# 可视化：每类一条示例
fig, axes = plt.subplots(4, 1, figsize=(16, 9), sharex=True)
for idx, (ax, class_name) in enumerate(zip(axes, classes)):
    example = signals[idx * n_per_class]
    ax.plot(t[:300], example[:300], linewidth=0.6, color='steelblue')
    ax.set_ylabel('值')
    ax.set_title(f'类别: {class_name}')
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel('时间 (s)')
fig.suptitle('四类信号示例 — 每类一条', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch13_fig1_signal_classes.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch13_fig1_signal_classes.png")

# ============================================================
# Step 2: 滑动窗口特征提取
# ============================================================
print("\n" + "=" * 60)
print("Step 2: 滑动窗口特征提取 — 时域 + 频域 + 峰特征")
print("=" * 60)

window_size = 256
stride = 128


def extract_features_from_window(w, fs_local):
    """从一个窗口提取全部特征。"""
    feat = {}
    N = len(w)

    # ---- 时域特征 ----
    feat['mean'] = np.mean(w)
    feat['std'] = np.std(w)
    w_centered = w - feat['mean']
    sigma = feat['std']
    if sigma > 1e-10:
        feat['skewness'] = np.mean(w_centered**3) / (sigma**3)
        feat['kurtosis'] = np.mean(w_centered**4) / (sigma**4)
    else:
        feat['skewness'] = 0.0
        feat['kurtosis'] = 0.0
    feat['rms'] = np.sqrt(np.mean(w**2))
    feat['crest_factor'] = (np.max(np.abs(w)) / feat['rms']
                            if feat['rms'] > 0 else 0)

    # ---- 频域特征 ----
    freqs, psd = welch(w, fs=fs_local, nperseg=min(64, N//4))
    total_power = np.sum(psd)
    if total_power > 1e-15:
        feat['spectral_centroid'] = np.sum(freqs * psd) / total_power
        # 低频能量比 (0-2Hz)
        low_mask = freqs <= 2
        feat['low_freq_ratio'] = np.sum(psd[low_mask]) / total_power
        # 频谱熵
        psd_norm = psd / total_power
        psd_norm = psd_norm[psd_norm > 0]
        feat['spectral_entropy'] = -np.sum(psd_norm * np.log(psd_norm))
    else:
        feat['spectral_centroid'] = 0.0
        feat['low_freq_ratio'] = 0.0
        feat['spectral_entropy'] = 0.0

    feat['dominant_freq'] = freqs[np.argmax(psd)] if total_power > 1e-15 else 0.0

    # ---- 峰特征 ----
    prom_threshold = max(0.2, sigma * 0.5)
    peaks, props = find_peaks(w, prominence=prom_threshold)
    feat['n_peaks'] = len(peaks)
    if len(peaks) > 0:
        feat['mean_prominence'] = np.mean(props['prominences'])
        feat['max_prominence'] = np.max(props['prominences'])
        feat['peak_density'] = len(peaks) / N
    else:
        feat['mean_prominence'] = 0.0
        feat['max_prominence'] = 0.0
        feat['peak_density'] = 0.0

    # ---- 包络特征 (Hilbert) ----
    analytic = hilbert(w)
    envelope = np.abs(analytic)
    feat['envelope_mean'] = np.mean(envelope)
    feat['envelope_std'] = np.std(envelope)

    return feat


# 对所有信号提取特征
feature_names = None
X_raw = []
y_raw = []

for i in range(n_total):
    sig = signals[i]
    for start in range(0, N_per_signal - window_size, stride):
        window = sig[start:start + window_size]
        feats = extract_features_from_window(window, fs)
        if feature_names is None:
            feature_names = list(feats.keys())
        X_raw.append([feats[k] for k in feature_names])
        y_raw.append(labels[i])

X_raw = np.array(X_raw)
y_raw = np.array(y_raw)

print(f"  特征矩阵: {X_raw.shape[0]} 窗口 × {X_raw.shape[1]} 特征")
print(f"  特征列表: {feature_names}")

# ============================================================
# Step 3: 标准化 + 训练/测试划分
# ============================================================
print("\n" + "=" * 60)
print("Step 3: 标准化 + 训练/测试划分")
print("=" * 60)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_raw, test_size=0.25, random_state=42, stratify=y_raw
)
print(f"  训练集: {X_train.shape[0]} 样本, 测试集: {X_test.shape[0]} 样本")

# ============================================================
# Step 4: 训练 Random Forest + 评估
# ============================================================
print("\n" + "=" * 60)
print("Step 4: Random Forest 分类 + 评估")
print("=" * 60)

clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = np.mean(y_pred == y_test)

print(f"\n  测试准确率: {accuracy:.4f}")
print(f"\n  分类报告:")
print(classification_report(y_test, y_pred, target_names=classes))

# 混淆矩阵
fig, ax = plt.subplots(figsize=(8, 7))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_names=classes).plot(
    ax=ax, cmap='Blues', values_format='d')
ax.set_title(f'混淆矩阵 (准确率={accuracy:.3f})')
plt.tight_layout()
plt.savefig('code/ch13_fig2_confusion.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch13_fig2_confusion.png")

# ============================================================
# Step 5: 特征重要性分析
# ============================================================
print("\n" + "=" * 60)
print("Step 5: 特征重要性 — 哪些信号特征最有区分力？")
print("=" * 60)

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

print(f"\n  Top-10 最重要特征:")
for rank, idx in enumerate(indices[:10]):
    print(f"    {rank+1}. {feature_names[idx]:<25s} = {importances[idx]:.4f}")

# 特征重要性图
fig, ax = plt.subplots(figsize=(12, 6))
ax.barh(range(len(indices)), importances[indices], color='steelblue')
ax.set_yticks(range(len(indices)))
ax.set_yticklabels([feature_names[i] for i in indices])
ax.invert_yaxis()
ax.set_xlabel('特征重要性')
ax.set_title('Random Forest 特征重要性排序')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('code/ch13_fig3_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch13_fig3_feature_importance.png")

# ============================================================
# Step 6: 特征分布可视化 — 两类对比
# ============================================================
print("\n" + "=" * 60)
print("Step 6: 特征分布 — 正常 vs 异常在关键特征上的差异")
print("=" * 60)

# 选取 top-4 特征，画分布对比
top4_indices = indices[:4]
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

for ax, fi in zip(axes.flat, top4_indices):
    fname = feature_names[fi]
    for class_idx, class_name in enumerate(classes):
        mask = y_raw == class_idx
        ax.hist(X_scaled[mask, fi], bins=40, density=True, alpha=0.5,
                label=class_name)
    ax.set_xlabel(f'{fname} (标准化后)')
    ax.set_ylabel('密度')
    ax.set_title(f'{fname} — 四类的分布')
    ax.legend(fontsize=7)

fig.suptitle('Top-4 特征在四个类别上的分布 — 分离度越高=该特征越重要',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('code/ch13_fig4_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 已保存 ch13_fig4_feature_distributions.png")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("🏁 第13章 — 全书完结！")
print("=" * 60)
print(f"""
📋 本章产出：

  笔记：doc/13_信号特征工程与ML集成.md
  代码：code/ch13_信号特征工程与ML集成.py
  图片：
    ch13_fig1_signal_classes.png        — 四类信号波形示例
    ch13_fig2_confusion.png            — 混淆矩阵
    ch13_fig3_feature_importance.png   — 特征重要性排序
    ch13_fig4_feature_distributions.png — Top-4特征分布对比

  📊 最终结果：
    测试准确率: {accuracy:.4f}
    特征总数: {len(feature_names)} (时域 {sum(1 for n in feature_names if n in ['mean','std','skewness','kurtosis','rms','crest_factor'])} + 频域 {sum(1 for n in feature_names if 'spectral' in n or 'freq' in n or 'dominant' in n)} + 峰 {sum(1 for n in feature_names if 'peak' in n or 'prominence' in n)}+ 包络 {sum(1 for n in feature_names if 'envelope' in n)})

🎯 全书核心收获：
  1. 信号 ≠ 表格数据 — 它的结构在频率和波形中
  2. 滤波 = 按频率分离成分（趋势/周期/噪声各自分析）
  3. 频谱 = "看见"数据隐藏的周期性
  4. 时频分析 = 追踪频率如何随时间变化
  5. 峰值检测 = 从连续信号到离散事件
  6. 去噪 = 保留结构、去掉偏离——同一种思维贯穿前处理和后处理
  7. 信号特征 = 从信号的动态模式中提取信息，而非从静态属性

  这些方法的共同哲学：
    "真实世界有结构。观测 = 结构 + 偏离。你的任务是提取结构。"
""")
