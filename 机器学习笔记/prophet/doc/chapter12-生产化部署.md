# 第 12 章：生产化部署

模型在 Jupyter 里跑得再好，最终要上线。这一章讲如何把 Prophet 从笔记本搬到生产环境——持久化、调度、监控、更新。

---

## 12.1 模型持久化

### 12.1.1 序列化方式

Prophet 模型支持三种持久化方式：

| 方式 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| **pickle / joblib** | 内部使用、快速原型 | 简单、快 | Python 版本/库版本绑定 |
| **JSON (模型参数)** | 跨平台、长期存储 | 可读、跨语言 | 只存参数，不存完整 Stan 对象 |
| **prophet.serialize** | Prophet 内置 | 官方支持、处理特殊对象 | 基于 JSON，文件可能很大 |

```python
import json
from prophet.serialize import model_to_json, model_from_json

# 保存
with open('model.json', 'w') as f:
    json.dump(model_to_json(model), f)

# 加载
with open('model.json', 'r') as f:
    model = model_from_json(json.load(f))
```

---

## 12.2 预测管道的设计

一个生产级预测管道通常包含以下步骤：

```
┌─────────────────────────────────────────────────────┐
│                    预测管道                          │
│                                                      │
│  数据库 ──→ 数据提取 ──→ 质量检查 ──→ 预处理         │
│                                              │       │
│                                              ▼       │
│  告警 ←── 异常检测 ←── 预测输出 ←── 模型拟合         │
│                                              │       │
│  数据库 ←── 结果写入 ←── 合理性检查 ←────────┘       │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### 12.2.1 基础管道代码

```python
import pandas as pd
from prophet import Prophet
import json
from prophet.serialize import model_to_json, model_from_json
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProphetPipeline:
    """生产级 Prophet 预测管道"""

    def __init__(self, model_params=None, model_path=None):
        self.model_params = model_params or {}
        self.model_path = model_path
        self.model = None

    def extract_data(self, query_or_df):
        """从数据源获取数据"""
        if isinstance(query_or_df, pd.DataFrame):
            df = query_or_df.copy()
        else:
            # 假设是 SQL 查询，用你的数据库连接执行
            # df = pd.read_sql(query_or_df, your_db_connection)
            raise NotImplementedError("请接入你的数据库连接")

        df['ds'] = pd.to_datetime(df['ds'])
        return df[['ds', 'y']].sort_values('ds')

    def check_quality(self, df):
        """数据质量检查"""
        checks = {
            'min_rows': len(df) >= 30,
            'no_all_nan': df['y'].notna().any(),
            'positive_variance': df['y'].std() > 0,
            'date_sorted': df['ds'].is_monotonic_increasing,
        }
        failed = [k for k, v in checks.items() if not v]
        if failed:
            raise ValueError(f"数据质量检查失败: {failed}")
        logger.info(f"数据质量检查通过, {len(df)} 行")

    def fit_or_load(self, df):
        """拟合新模型或加载已有模型"""
        if self.model_path:
            try:
                with open(self.model_path, 'r') as f:
                    self.model = model_from_json(json.load(f))
                logger.info(f"从 {self.model_path} 加载模型")
                return
            except FileNotFoundError:
                logger.info("模型文件不存在，重新训练")

        self.model = Prophet(**self.model_params)
        self.model.fit(df)
        logger.info("模型训练完成")

    def predict(self, periods=90, freq='D'):
        """生成预测"""
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)
        return forecast

    def check_reasonableness(self, forecast, historical):
        """合理性检查"""
        recent_mean = historical['y'].tail(30).mean()
        forecast_mean = forecast['yhat'].tail(periods).mean()

        # 检查未来预测是否与最近历史严重偏离
        ratio = forecast_mean / recent_mean if recent_mean > 0 else 1
        if ratio > 3 or ratio < 0.33:
            logger.warning(f"预测均值 ({forecast_mean:.0f}) 偏离最近均值 ({recent_mean:.0f}) {ratio:.1f}倍")

        # 检查是否有不合理的负值
        if historical['y'].min() >= 0 and (forecast['yhat'] < 0).any():
            logger.warning("预测含负值（历史无非负值）")

    def save_model(self):
        """保存模型"""
        if self.model and self.model_path:
            with open(self.model_path, 'w') as f:
                json.dump(model_to_json(self.model), f)
            logger.info(f"模型已保存到 {self.model_path}")

    def run(self, data_source, periods=90):
        """完整管道"""
        # Step 1: 提取
        df = self.extract_data(data_source)

        # Step 2: 质量检查
        self.check_quality(df)

        # Step 3: 拟合/加载
        self.fit_or_load(df)

        # Step 4: 预测
        forecast = self.predict(periods)

        # Step 5: 合理性检查
        self.check_reasonableness(forecast, df)

        # Step 6: 保存
        self.save_model()

        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# 使用示例
pipeline = ProphetPipeline(
    model_params={'yearly_seasonality': True, 'weekly_seasonality': True},
    model_path='production_model.json',
)
forecast = pipeline.run(df, periods=90)
```

---

## 12.3 自动重训练策略

模型不会永远有效——新数据到来后应该更新。

### 策略对比

| 策略 | 触发条件 | 优点 | 缺点 |
|------|---------|------|------|
| **定时重训练** | 每天/每周定时 | 简单、可预测 | 不感知数据变化 |
| **误差触发** | 最近预测误差超过阈值 | 自适应 | 需要持续评估 |
| **数据量触发** | 新数据累积 N 天后 | 平衡计算资源 | 可能延迟 |
| **全量重训练** | 每次预测前 | 最新 | 大数据时很慢 |

### 推荐组合

```python
def should_retrain(last_train_time, error_monitor, config):
    """判断是否需要重新训练"""
    checks = []

    # 1. 距上次训练超过 7 天
    if last_train_time:
        days_since = (datetime.now() - last_train_time).days
        checks.append(days_since > config.get('max_staleness_days', 7))

    # 2. 最近预测误差超过阈值
    if error_monitor:
        recent_mape = error_monitor.get_recent_mape(window=7)
        checks.append(recent_mape > config.get('mape_threshold', 20))

    # 3. 新数据累积超过阈值
    new_data_count = get_new_data_count_since(last_train_time)
    checks.append(new_data_count > config.get('new_data_threshold', 30))

    return any(checks)
```

---

## 12.4 预测监控

### 上线的模型需要持续监控三个维度

```
1. 预测精度监控
   ├── 每个预测周期结束后，对比实际值
   ├── 指标：MAPE、RMSE 的分周期趋势
   └── 触发告警：精度持续恶化

2. 数据漂移监控
   ├── 输入数据的分布是否发生了变化
   ├── 指标：均值、方差、缺失率的变化
   └── 触发告警：分布大幅偏移

3. 模型健康监控
   ├── 模型是否还在生成合理预测
   ├── 指标：预测区间宽度、负值率、趋势方向
   └── 触发告警：预测值不合常理
```

```python
class ForecastMonitor:
    """预测效果监控"""

    def __init__(self):
        self.predictions = []  # 存历史预测
        self.actuals = []      # 存实际值

    def log_prediction(self, ds, yhat, yhat_lower, yhat_upper):
        self.predictions.append({
            'ds': ds, 'yhat': yhat,
            'yhat_lower': yhat_lower, 'yhat_upper': yhat_upper,
        })

    def log_actual(self, ds, actual):
        self.actuals.append({'ds': ds, 'actual': actual})

    def check(self):
        """检查最近的预测表现"""
        if len(self.actuals) < 7:
            return

        recent = pd.DataFrame(self.actuals[-7:])
        recent_pred = pd.DataFrame(self.predictions[-7:])

        merged = recent.merge(recent_pred, on='ds')
        errors = merged['actual'] - merged['yhat']

        mape = (abs(errors) / merged['actual']).mean() * 100
        coverage = ((merged['actual'] >= merged['yhat_lower']) &
                    (merged['actual'] <= merged['yhat_upper'])).mean()

        alerts = []
        if mape > 30:
            alerts.append(f"⚠️  MAPE 过高: {mape:.1f}%")
        if coverage < 0.5:
            alerts.append(f"⚠️  区间覆盖率过低: {coverage:.1%}")

        return {'mape': mape, 'coverage': coverage, 'alerts': alerts}
```

---

## 12.5 CICD 与调度的精简方案

对于小团队，不需要 Kubernetes。实用方案：

```bash
# crontab 示例（Linux/macOS）
# 每天早上 6 点运行预测管道

0 6 * * * cd /path/to/prophet_project && uv run python pipeline.py >> logs/predict.log 2>&1
```

```python
# pipeline.py — 生产脚本入口
import logging
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f'logs/pipeline_{datetime.now():%Y%m%d}.log'),
        logging.StreamHandler(sys.stdout),
    ]
)

def main():
    logger = logging.getLogger(__name__)
    logger.info("开始预测管道")

    # ... 数据提取、模型拟合、预测、写入数据库 ...

    logger.info("预测管道完成")

if __name__ == '__main__':
    main()
```

---

## 12.6 常见问题速查

| 问题 | 原因 | 解决 |
|------|------|------|
| pickle 加载失败 | Python/Prophet 版本不匹配 | 用 JSON 序列化替代 pickle |
| 模型文件巨大（>100MB） | Stan 编译信息被打包 | 使用 `model_to_json` 而非全量序列化 |
| 管道定时任务失败无感知 | 没有监控 | 加日志 + 失败告警（企业微信/邮件） |
| 新数据格式变化 | 上游数据源变更 | 在 `extract_data` 阶段做强校验 |
| 并发预测竞争 | 多进程同时写同一模型文件 | 用文件锁或数据库管理模型版本 |

---

## 12.7 核心概念清单

| 概念 | 一句话理解 |
|------|-----------|
| **模型持久化** | JSON 序列化 > pickle（跨版本更安全） |
| **预测管道** | 提取 → 检查 → 训练 → 预测 → 合理性验证 → 写入 |
| **自动重训练** | 按时间 + 误差 + 数据量三个维度触发 |
| **监控三件套** | 精度监控 + 数据漂移 + 模型健康 |
| **crontab** | 小团队的生产调度方案——简单但有效 |

---

下一章是最后的实战案例——把前面 12 章的知识糅合进四个真实场景。
