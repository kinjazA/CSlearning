# 第14章 · tsfresh 自动特征工程

---

## 0. 这一章在全书中的位置

### 0.1 你不是刚刚学完信号特征才来学 tsfresh

顺序是这样的：

1. 第 1~12 章：你学会了信号处理的**底层工具**（滤波、频谱、去噪、峰值检测、Hilbert……）
2. 第 13 章：你学会了**手动**从信号中提取特征（时域/频域/峰/包络）
3. **本章**：你学会用 tsfresh **自动化**这个过程——一行代码提取 1200+ 个特征，再用统计检验筛选出真正有用的

**关键认知**：tsfresh 不会替代你前面学的内容。它依赖你前面学的内容来：
- 判断提取出的特征是否可信（比如你知道 Nyquist 限制，就能判断某个频域特征是否被混叠污染）
- 在特征筛选后，理解为什么某些特征被选中（比如 `fft_aggregated` 的某类统计量被选中 → 你知道是频谱形状包含了区分信息）
- 当 tsfresh 的特征不够用时，自己动手加自定义特征

### 0.2 前13章和本章的关系

| 前 13 章 | 本章 (tsfresh) |
|----------|---------------|
| 你知道怎么设计滤波器 | tsfresh 不知道有没有混叠——你来判断 |
| 你知道怎么检测峰值 | tsfresh 能算峰的数量——但参数你调 |
| 你知道频谱分辨率 ≈ fs/N | tsfresh 不做这个判断——你来定窗口长度 |
| 你手动提取 16 个特征 | tsfresh 自动生成 1200+ 个特征 |
| 你凭经验选特征 | tsfresh 用 Benjamini-Hochberg 统计检验筛选 |

---

## 1. tsfresh 是什么？

### 1.1 一句话

> **tsfresh** = 对时间序列**自动**提取数百个特征 + 用假设检验自动筛选出对目标变量有显著区分力的特征。

你第 13 章手动写了 `extract_signal_features()`，提取了 16 个特征。tsfresh 做的是同一件事——但它把一个信号拆成几百种不同的统计量，每种统计量可以从多个角度计算，最终产生一个 1200+ 维的特征向量。

### 1.2 它和 scipy.signal 的分工

```
scipy.signal（你已掌握）          tsfresh（本章）
─────────────────────────        ─────────────────
底层信号处理操作                  高层特征自动化
  - 滤波/去噪/重采样                - extract_features()
  - 频谱/时频分析                   - select_features()
  - 峰值检测                       - relevance_table()
  - Hilbert / 相干性               - feature filtering
        ↓                              ↓
  对信号"做手术"                  对信号"做体检报告"
```

**正确的协作方式**：
```python
# 1. scipy 做预处理（你前面学的全部用在这里）
clean = preprocessor.process(raw_signal)    # 去趋势、去噪、重采样

# 2. tsfresh 在干净的信号上提取特征
features = extract_features(...)

# 3. 你看 tsfresh 的输出 → 用你的信号知识理解"为什么这个特征重要"
```

---

## 2. 数据格式要求

### 2.1 tsfresh 需要的列格式

tsfresh 要求输入 DataFrame，包含三列（名字可以自定，但逻辑必须对应）：

| 列 | 含义 | 例子 |
|----|------|------|
| `id` | 每条独立时间序列的唯一标识 | `"stock_000001"` / `"sensor_A"` / `"day_2024-03-15"` |
| `time` | 时间索引 | `0, 1, 2, ...` 或 `datetime` |
| `value` | 观测值 | `10.5, 10.8, 10.3, ...` |

### 2.2 数据形状

```
原始数据（每条信号一行）：
  stock_A: [10.2, 10.5, 10.3, 10.8, ...]
  stock_B: [25.1, 24.8, 25.3, 25.0, ...]

转成 tsfresh 需要的格式（每行 = 一个时刻的一个值）：
  id       time  value
  stock_A  0     10.2
  stock_A  1     10.5
  stock_A  2     10.3
  ...
  stock_B  0     25.1
  stock_B  1     24.8
  ...
```

### 2.3 转换代码

```python
import pandas as pd
import numpy as np

def signals_to_tsfresh_format(signals_dict, time_name='time', value_name='value'):
    """
    把 {id: array} 转成 tsfresh 的 (id, time, value) 格式。

    Parameters
    ----------
    signals_dict : dict
        key = 信号ID, value = 一维 numpy array

    Returns
    -------
    pd.DataFrame : 列 [id, time, value]
    """
    records = []
    for sig_id, values in signals_dict.items():
        for t, v in enumerate(values):
            records.append({'id': sig_id, time_name: t, value_name: v})
    return pd.DataFrame(records)

# 用法
signals = {'sensor_01': np.random.randn(500),
           'sensor_02': np.random.randn(500)}
df = signals_to_tsfresh_format(signals)
```

如果你的数据已经按时间自然有序（比如每天一行），可以不需要额外的 time 列，tsfresh 会按出现顺序处理。

---

## 3. 特征提取：`extract_features`

### 3.1 基础用法

```python
from tsfresh import extract_features

# df: 列 = [id, time, value]
# column_id: 标识符列名
# column_sort: 排序依据列名（通常是 time）

features = extract_features(df,
                            column_id='id',
                            column_sort='time',
                            column_value='value')
```

返回一个 DataFrame：
- 行 = 每条原始信号（按 `id`）
- 列 = 每个特征（可能有 700~1200+ 列，取决于参数）

### 3.2 特征量的控制

默认提取所有可能的特征（~1200 列）。如果你的数据量大，可能非常慢。tsfresh 提供了三级缩减：

```python
from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters, EfficientFCParameters

# 方式1：使用预设的特征集

# Minimal — 仅最基础的特征，约 10 个（mean, variance, 分位数等）
features = extract_features(df, column_id='id', column_sort='time',
                            default_fc_parameters=MinimalFCParameters())

# Efficient — 计算效率较高的特征，约 100+ 个
features = extract_features(df, column_id='id', column_sort='time',
                            default_fc_parameters=EfficientFCParameters())

# Comprehensive — 全部特征（默认），约 1200 个
features = extract_features(df, column_id='id', column_sort='time',
                            default_fc_parameters=ComprehensiveFCParameters())
```

```python
# 方式2：手动指定想要哪些类别的特征
from tsfresh.feature_extraction import extract_features
from tsfresh.feature_extraction.settings import from_columns

# 第一步：用 Minimal 提取 → 看到所有特征列名
features_mini = extract_features(df, column_id='id', column_sort='time',
                                 default_fc_parameters=MinimalFCParameters())
# 第二步：看列名，决定要加哪些 → 构建自定义参数
# （高级用法，需要查阅 tsfresh 文档的 feature 名称）
```

### 3.3 n_jobs 并行加速

```python
# tsfresh 内部大量循环计算，并行可以显著加速
features = extract_features(df, column_id='id', column_sort='time',
                            n_jobs=4)   # 使用 4 个 CPU 核心
```

### 3.4 按信号分段提取（大数据的正确做法）

如果你的信号巨长，可以**先切成窗口**，再对每个窗口提取特征：

```python
# 每条长信号切为多个固定长度的段，每段独立提取特征
def extract_features_in_windows(signal, window_size, stride, fs, signal_id):
    """滑动窗口 + tsfresh 特征提取。返回 DataFrame。"""
    frames = []
    for i, start in enumerate(range(0, len(signal) - window_size, stride)):
        window = signal[start:start + window_size]
        df_seg = pd.DataFrame({
            'id': [f'{signal_id}_win{i}'],    # 每个窗口一个唯一 ID
            'time': range(len(window)),
            'value': window
        })
        frames.append(df_seg)
    df_all = pd.concat(frames, ignore_index=True)
    return extract_features(df_all, column_id='id', column_sort='time')

# 这和你在第13章做的滑动窗口特征提取是同一件事，只是提取器换成了 tsfresh
```

---

## 4. 特征筛选：`select_features`（核心价值）

这是 tsfresh 最有价值的功能——**自动判断哪些特征对分类/回归任务是真正有用的。**

### 4.1 问题

你提取了 1200 个特征，但你只有 200 条信号。直接扔进 RF/XGBoost → 必定过拟合。而且绝大多数特征对你要预测的目标变量没有任何区分力。

tsfresh 的做法：**对每个特征，做一个统计检验，看它和标签是否独立。** 如果独立 → 这个特征没用，扔掉。如果不独立 → 保留。同时用 Benjamini-Hochberg 程序纠正多重检验的假阳性。

### 4.2 基础用法

```python
from tsfresh import select_features

# features: extract_features() 的输出（行=样本，列=特征）
# y: 标签 Series，index 必须和 features 的 index 对应
#     对于分类：y 是类别值（int/str）
#     对于回归：tsfresh 内部做离散化再检验

X_filtered = select_features(features, y)
```

返回的 `X_filtered` 是一个缩减后的 DataFrame：只保留了那些和标签有显著关联的特征。列数从 1200+ 降到几十到几百（取决于问题的信号有多复杂）。

### 4.3 统计原理（和人话翻译）

tsfresh 为每个特征做如下操作：

1. **如果 y 是离散标签（分类）**：对每个特征做 Kruskal-Wallis H 检验（一种非参数 ANOVA）或 Mann-Whitney U 检验（两组对比）。**翻译**：这个特征在类别 A 和类别 B 上的分布是否显著不同？如果是 → 保留（特征能区分不同类别）。

2. **如果 y 是连续值（回归）**：tsfresh 先将 y 离散化成若干个 bins，再做卡方检验。**翻译**：这个特征在不同的 y 取值区间上是否有不同的分布？

3. **多重检验矫正**：你在同时检验 1200 个假设（每个特征一个"和标签无关"的零假设）。即使所有特征都是随机噪声，按 α=0.05 你也会有约 60 个假阳性。Benjamini-Hochberg 程序控制**假发现率（FDR）**——在"宣称显著"的特征中，假阳性比例不超过你设定的阈值（默认 5%）。

### 4.4 获取每个特征的显著性

```python
from tsfresh import relevance_table

# 看每个特征和标签的关联有多强
rel = relevance_table(features, y)
# 列: feature_name, p_value, relevant (True/False), type

# 只看被判定为"显著"的特征
significant = rel[rel['relevant'] == True]
# 按 p 值排序
significant = significant.sort_values('p_value')
```

### 4.5 fdr_level 参数

```python
# 放宽筛选 → 保留更多特征（但假阳性比例更高）
X_filtered = select_features(features, y, fdr_level=0.10)   # FDR ≤ 10%

# 收紧筛选 → 只保留最可靠的（假阳性比例 ≤ 1%）
X_filtered = select_features(features, y, fdr_level=0.01)
```

---

## 5. tsfresh 的特征分类：它算了什么？

### 5.1 特征大类速查

tsfresh 把 1200+ 个特征组织为"**对信号做某种操作 → 再对这个操作的结果计算各种统计量**"的结构：

| 信号操作 | 功能 | 对应你学过的章节 |
|---------|------|:---:|
| **原始信号** | 对原始值直接计算均值/方差/偏度/峭度/分位数…… | 第 1、2 章 |
| **一阶差分** | 对变化量（相当于收益率）计算统计量 | 第 2 章差分核 |
| **二阶差分** | 对加速度计算统计量 | 第 2 章 |
| **FFT 系数** | 对傅里叶变换的系数（实部/虚部/相位/幅度）算统计量 | **第 7 章** |
| **FFT 聚合带** | 将频谱分成 5 个频带，每个频带算能量中心/方差/偏度/峭度/过零率 | **第 7 章** |
| **小波系数** | 离散小波分解后各层系数的统计量 | 第 8 章 |
| **自相关** | ACF 不同 lag 上的值/聚合统计量 | 第 2 章 |
| **AR 系数** | 用 Burg 方法拟合 AR 模型的系数（谱估计） | 第 7 章 |
| **CID/CE** | 复杂度不变距离 → 测量信号的复杂性 | 非线性特征 |
| **峰检测** | 峰的数量/间距/突出度统计量 | **第 9 章** |
| **近似熵/样本熵** | 信号的不可预测程度 | 第 13 章非线性 |
| **矩阵分布** | 将信号分割成 bin，横向/纵向统计量 | 时域结构 |

### 5.2 命名规则

tsfresh 的特征列名由三部分组成：`<操作名>__<统计量>`

```
例如:  fft_aggregated__variance          → "频域聚合特征的方差"
       absolute_sum_of_changes__mean     → "一阶差分绝对值的均值"
       cwt_coefficients__variance        → "连续小波系数的方差"
```

双下划线 `__` 分隔"做了什么操作"和"算的什么统计量"。

### 5.3 和前 13 章的对应关系

当你看到 tsfresh 输出 `fft_coefficient__coeff_0__attr_"abs"` 被 `select_features` 选中时——你知道这意味着"频谱的第一个系数的幅度在类别间有显著差异"，对应你第 7 章学的主导频率分析。

当你看到 `number_peaks__n_peaks` 被选中时——你知道这意味着"信号中事件密度在类别间不同"，对应你第 9 章的峰值检测。

**你不只是"看到这个特征被选中"，而是能理解它在物理上意味着什么。** 这就是前 13 章的价值——它们让你成为特征筛选结果的**解释者**，而不是只能盲目信任一个黑箱输出的操作员。

---

## 6. 完整端到端流程

### 6.1 代码模板

```python
import pandas as pd
import numpy as np
from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import EfficientFCParameters
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ====== Step 1: 准备数据 ======
# 假设你已经有:
#   signals_list: list of 1D numpy arrays  (每条一个 array)
#   labels_list:  list of int/str           (每条一个标签)
#   signal_ids:   list of str               (每条一个 ID)

# 转成 tsfresh 格式
records = []
for sig_id, sig_vals in zip(signal_ids, signals_list):
    for t, v in enumerate(sig_vals):
        records.append({'id': sig_id, 'time': t, 'value': v})
df = pd.DataFrame(records)

y = pd.Series(labels_list, index=signal_ids)   # index = id

# ====== Step 2: 提取特征 ======
print("提取特征中...")
X_raw = extract_features(df, column_id='id', column_sort='time',
                         default_fc_parameters=EfficientFCParameters(),
                         n_jobs=4)
print(f"  原始特征数: {X_raw.shape[1]}")

# ====== Step 3: 筛选显著特征 ======
print("筛选中...")
X_selected = select_features(X_raw, y, fdr_level=0.05)
print(f"  筛选后特征数: {X_selected.shape[1]}")

# ====== Step 4: 训练模型 ======
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.25, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f"\n准确率: {np.mean(y_pred == y_test):.4f}")
print(classification_report(y_test, y_pred))

# ====== Step 5: 查看哪些特征最重要 ======
importances = clf.feature_importances_
top_idx = np.argsort(importances)[::-1][:15]
print("\nTop-15 最重要的特征:")
for idx in top_idx:
    print(f"  {X_selected.columns[idx]:<50s}  {importances[idx]:.4f}")
```

### 6.2 关键注意事项

**1. `y` 的 index 必须和 `features` 的 index 对齐**

```python
# y 的 index = 信号的 id（和 features 的行 index 一致）
y = pd.Series([0, 1, 0, 1, ...], index=['signal_0', 'signal_1', ...])
```

如果不对齐，`select_features` 会报错。

**2. 训练/测试必须先划分再筛选**

```python
# ❌ 错误：先筛选再划分 → 数据泄漏
X_selected = select_features(X_raw, y)
X_train, X_test, y_train, y_test = train_test_split(X_selected, y)

# ✅ 正确：先划分，再在训练集上筛选，测试集复用筛选结果
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y)
X_train = select_features(X_train_raw, y_train)
# 测试集只保留和训练集相同的列
X_test = X_test_raw[X_train.columns]
```

**为什么？** 如果你在全部数据上筛选特征——筛选过程"看到"了测试集 → 信息泄漏 → 你在测试集上的准确率被高估。

**3. 相关性去重**

`select_features` 只做了"特征和标签的独立性检验"。如果有两个高度相关的特征都被选中（比如 `fft_coefficient__coeff_0__attr_"abs"` 和 `fft_aggregated__variance` 可能高度相关），你需要额外做去重：

```python
# 在 X_selected 上做相关性过滤
corr_matrix = X_selected.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
X_final = X_selected.drop(columns=to_drop)
```

---

## 7. 什么时候用 tsfresh，什么时候自己手动提特征？

| 场景 | 用 tsfresh | 手动提取 |
|------|:---:|:---:|
| 你不确定哪些特征有用，想快速探索 | ✅ | |
| 数据量大（很多条信号），自动筛选省时间 | ✅ | |
| 要做 baseline，快速建立一个可用的模型 | ✅ | |
| 你对特征有明确的物理/业务解释需求 | | ✅ |
| 你需要稳定的生产级特征集（不想每次跑自动筛选） | | ✅ |
| 信号很短（<50 点），很多 tsfresh 特征算不出来或不可靠 | | ✅ |
| 需要实时/边缘计算（特征计算必须在低资源环境跑） | | ✅ |
| 你的特征空间需要基于领域知识自定义 | | ✅ |

**最务实的策略**：

1. **探索阶段**：用 tsfresh 跑一遍，看哪些特征被选中 → 理解"什么样的信号结构是关键"
2. **生产阶段**：手动实现被选中的那 10~20 个核心特征，去掉依赖 tsfresh 的整库依赖，同时保证可解释性
3. **混合阶段**：`tsfresh` 无法覆盖的特定领域特征（如 Hilbert 瞬时频率的稳定性、相干性的时间演化）→ 自己加进特征集

---

## 8. tsfresh 的局限——你应该知道的事

### 8.1 它不理解信号处理的物理含义

tsfresh 不知道 Nyquist 频率。你给它日度数据，它照样提取 `fft_coefficient` 特征——如果存在混叠，这些特征就是垃圾，但 tsfresh 不会告诉你。

**你的角色**：在把数据送入 tsfresh 之前，用你前 12 章的知识确保数据是"干净且采样正确"的。

### 8.2 统计显著 ≠ 业务有用

`select_features` 的筛选结果是统计上显著的——但不代表这个特征在业务上是最优的。一个 p=0.0001 的特征和标签的关联可能是非线性的、不稳定的、样本依赖的。

**你的角色**：看你被选中的特征的列名，用信号知识判断它们是否"合理"。如果一个特征在物理上不应该和你的目标变量有关（比如你预测的是日度波动率，但 `fft_coefficient` 的最高频被选中了——检查是不是混叠搞的鬼），你要能意识到并质疑。

### 8.3 短序列不可靠

tsfresh 很多特征（特别是 AR 系数、FFT 聚合带、样本熵）需要足够长的序列才能稳定估计。如果你的信号 < 100 个采样点，很多特征的计算结果不可靠。

**你的角色**：判断你的信号长度是否足够支撑这些特征——这是你第 7 章学的"频率分辨率 ≈ fs/N"直接告诉你的。

### 8.4 计算成本

Comprehensive 模式 1200+ 特征 × 几百条信号 = 可能需要几分钟甚至更长。对于日常探索，从 `EfficientFCParameters()` 开始，确认方向后再考虑加特征。

---

## 9. 一个具体的探索工作流

```
原始信号 (N条)
    │
    ├── [scipy.signal] 预处理管线（第11章）
    │    去趋势 → 抗混叠 → 去噪 → 标准化
    │
    ├── [tsfresh] extract_features(EfficientFCParameters)
    │    ~120 个特征 → DataFrame (N × 120)
    │
    ├── [tsfresh] select_features(y, fdr_level=0.05)
    │    筛选到 ~20-50 个显著特征
    │
    ├── [你] 看列名，判断是否有混叠/不可靠特征 → 手动剔除
    │
    ├── [sklearn] 训练 RF / XGBoost
    │
    ├── [你] 看 feature_importances_ 的 Top-10
    │    问自己："这些特征在物理上意味着什么？"
    │    如果不能回答 → 回到 scipy.signal 做诊断
    │
    └── 最终：选 10-15 个既显著又可解释的特征 → 作为生产特征集
```

---

## 10. 本章要点速查

| 概念 | 一句话 |
|------|--------|
| `extract_features` | 一行代码从所有信号提取 1200+ 特征 |
| 数据格式 | (id, time, value) 的长表格式 |
| `MinimalFCParameters` | 仅 10 个基础特征 — 最轻量 |
| `EfficientFCParameters` | ~120 个高效特征 — **推荐日常起点** |
| `ComprehensiveFCParameters` | 全部 1200+ 特征 — 探索用 |
| `select_features` | 统计检验 + Benjamini-Hochberg → 自动保显著特征 |
| FDR | False Discovery Rate — 宣称显著里面假阳性的预期比例 |
| `relevance_table` | 查看每个特征和标签的具体显著性 |
| 特征命名：`op__stat` | `fft_coefficient__variance` = FFT 系数的方差 |
| 数据泄漏 | 必须先划分再筛选，不能先筛选再划分 |
| tsfresh vs 手动 | 探索用 tsfresh，生产用手动实现的核心特征集 |

---

> **运行代码**：本章不提供独立 `.py` 代码文件——tsfresh 的用法在笔记各段已完整呈现，直接复制代码模板即可使用。
>
> **安装**：`pip install tsfresh`
>
> **回顾**：第13章是手动特征工程（scipy.signal 手写），本章是自动特征工程（tsfresh），两章互补。当你用 tsfresh 发现了一个被选中的特征但看不懂它是什么时，回到前 13 章的对应章节查它的物理含义。
