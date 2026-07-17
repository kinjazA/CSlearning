# 第 2 章：Prophet 快速入门

## 2.1 安装与环境配置

### 依赖关系

Prophet 的核心依赖是 **Stan**（贝叶斯推断引擎）。本课程使用 [uv](https://docs.astral.sh/uv/) 统一管理环境。

```bash
# 1. 安装 uv（如未安装）
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. 在项目根目录创建虚拟环境
uv venv --python 3.10

# 3. 激活环境
# macOS / Linux: source .venv/bin/activate
# Windows:       .venv\Scripts\activate

# 4. 安装依赖
uv pip install prophet pandas matplotlib scikit-learn jupyter

# 5. 验证安装
uv run python -c "from prophet import Prophet; print('OK')"
```

### 日常运行

本课程所有脚本通过 `uv run python` 执行，确保始终使用项目虚拟环境：

```bash
# 运行指定脚本
uv run python code/chapter02/01_first_model.py

# 启动 Jupyter（如有 notebook）
uv run jupyter notebook
```

---

## 2.2 数据格式约定

Prophet 对输入 DataFrame 有严格的列名要求：

| 列名 | 含义 | 类型 | 是否必需 |
|------|------|------|----------|
| `ds` | 日期时间戳 | `datetime64` 或 `str` (YYYY-MM-DD 格式) | ✅ 必需 |
| `y` | 待预测的数值 | `float` | ✅ 必需 |
| `cap` | 饱和上限（仅逻辑增长） | `float` | 逻辑增长时必需 |
| `floor` | 饱和下限（仅逻辑增长） | `float` | 可选 |

```python
import pandas as pd

# 标准输入格式
df = pd.DataFrame({
    'ds': pd.date_range('2020-01-01', periods=365, freq='D'),
    'y': ...  # 你的数值序列
})
```

> ⚠️ **关键约定**：列名必须是 `ds` 和 `y`，大小写敏感。这是初学者最容易犯的错误。

---

## 2.3 核心 API 流程

Prophet 的设计遵循 scikit-learn 的风格，API 极为简洁：

```python
from prophet import Prophet

# Step 1: 创建模型实例
model = Prophet()

# Step 2: 拟合历史数据
model.fit(df)

# Step 3: 构建未来日期
future = model.make_future_dataframe(periods=365)  # 预测未来 365 天

# Step 4: 生成预测
forecast = model.predict(future)

# Step 5: 查看结果
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
```

### 核心方法一览

| 方法 | 作用 | 参数 |
|------|------|------|
| `Prophet(...)` | 创建模型，指定所有超参数 | 见后续章节 |
| `.fit(df)` | 用历史数据拟合模型 | `df` 必须有 `ds`, `y` 列 |
| `.make_future_dataframe(periods, freq)` | 创建包含未来日期的 DataFrame | `periods` 预测期数，`freq` 频率（默认 'D'） |
| `.predict(future)` | 在 future 日期上生成预测 | 返回包含 `yhat` 等的 DataFrame |
| `.plot(forecast)` | 绘制预测图（matplotlib） | 返回 Figure 对象 |
| `.plot_components(forecast)` | 绘制成分分解图 | 趋势 + 各季节性 |

---

## 2.4 预测结果解读

`forecast` DataFrame 包含的列：

| 列名 | 含义 | 何时关注 |
|------|------|----------|
| `ds` | 日期 | 始终 |
| `yhat` | **点预测值** | 业务决策的核心依据 |
| `yhat_lower` | 预测下界 (默认 80% 区间) | 悲观场景 / 安全库存下限 |
| `yhat_upper` | 预测上界 (默认 80% 区间) | 乐观场景 / 容量规划上限 |
| `trend` | 趋势分量 | 理解长期走向 |
| `weekly` / `yearly` / `daily` | 各季节性分量 | 理解周期性模式 |
| `holidays` | 节假日效应分量 | 量化节假日影响 |
| `additive_terms` / `multiplicative_terms` | 加性 / 乘性总分量 | 成分归因分析 |

---

## 2.5 第一个完整示例

```python
# code/chapter02/01_first_model.py
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# ============================================================
# 1. 准备数据 — 经典航空乘客数据
# ============================================================
url = "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_air_passengers.csv"
df = pd.read_csv(url)
print(f"数据形状: {df.shape}")
print(f"日期范围: {df['ds'].min()} ~ {df['ds'].max()}")
print(df.head())

# ============================================================
# 2. 拟合模型
# ============================================================
model = Prophet(
    yearly_seasonality=True,   # 年季节性（傅里叶阶数默认 10）
    weekly_seasonality=True,   # 周季节性（傅里叶阶数默认 3）
    daily_seasonality=False,   # 日季节性（日均以上数据不需要）
)
model.fit(df)
print("\n模型拟合完成！")

# ============================================================
# 3. 生成未来预测
# ============================================================
future = model.make_future_dataframe(periods=365)  # 预测未来一年
forecast = model.predict(future)

# 只看未来部分的关键列
future_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10)
print("\n未来 10 天预测:")
print(future_forecast.to_string())

# ============================================================
# 4. 可视化
# ============================================================
fig1 = model.plot(forecast)
plt.title('Prophet Forecast — Airline Passengers')
plt.xlabel('Date')
plt.ylabel('Passengers')
plt.savefig('forecast_plot.png', dpi=150, bbox_inches='tight')
plt.close()

fig2 = model.plot_components(forecast)
plt.savefig('components_plot.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n图表已保存: forecast_plot.png, components_plot.png")
```

---

## 2.6 常见踩坑与注意事项

### 坑 1：列名大小写
```python
# ❌ 错误
df.columns = ['DS', 'Y']

# ✅ 正确
df.columns = ['ds', 'y']
```

### 坑 2：日期格式
```python
# ❌ 错误 — 字符串格式不一致
df['ds'] = ['2020/01/01', '2020-01-02', '01-03-2020']

# ✅ 正确 — 统一 datetime64
df['ds'] = pd.to_datetime(df['ds'])
```

### 坑 3：单变量输入
Prophet 的 `y` 列只能放一个数值变量。如需多序列预测，每个序列独立建模（见第 11 章）。

### 坑 4：数据量检查
```python
# 至少需要一个完整季节周期的数据
if len(df) < 365:  # 对于年季节性
    print("⚠️ 警告：数据量不足以可靠地估计年季节性")
```

### 坑 5：频率推断
```python
# make_future_dataframe 默认频率是 'D'（天）
# 如果数据是月度的，需指定 freq='MS'
future = model.make_future_dataframe(periods=12, freq='MS')
```

### 坑 6：Windows 时区问题
```python
# Windows 上可能遇到时区相关警告，建议：
df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
```

---

## 2.7 本章小结

| 要点 | 说明 |
|------|------|
| 核心流程 | `Prophet()` → `.fit(df)` → `.make_future_dataframe()` → `.predict()` → `.plot()` |
| 数据格式 | 必须 `ds` (datetime) + `y` (float) 两列 |
| 预测输出 | `yhat` (点预测) + `yhat_lower`/`yhat_upper` (区间) |
| 最小数据量 | ≥ 一个完整季节周期的数据量 |

下一章将深入趋势建模——Prophet 最核心的组件。
