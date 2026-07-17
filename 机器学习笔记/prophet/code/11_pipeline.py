"""
第 12 章：生产化部署
====================
- 模型序列化（JSON）
- ProphetPipeline 生产管道
- 自动重训练判断
- 预测监控
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

# ============================================================
# 0. 准备
# ============================================================
Path('logs').mkdir(exist_ok=True)
Path('models').mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

# 模拟历史数据
np.random.seed(42)
dates = pd.date_range('2022-01-01', datetime.now().strftime('%Y-%m-%d'), freq='D')
n = len(dates)
trend = np.linspace(100, 250, n)
weekly = 30 * np.sin(2 * np.pi * np.arange(n) / 7)
yearly = 40 * np.sin(2 * np.pi * np.arange(n) / 365.25)
noise = np.random.randn(n) * 10
df_production = pd.DataFrame({
    'ds': dates,
    'y': np.maximum(trend + weekly + yearly + noise, 10),
})
logger.info(f"模拟生产数据: {len(df_production)} 天")

# ============================================================
# 1. 模型序列化
# ============================================================
def demo_serialization():
    """演示 JSON 序列化与反序列化"""
    logger.info("=== 模型序列化演示 ===")

    # 训练
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(df_production.tail(365))  # 只用最近一年

    # 保存
    model_path = 'models/demo_model.json'
    with open(model_path, 'w', encoding='utf-8') as f:
        json.dump(model_to_json(model), f, ensure_ascii=False)
    logger.info(f"模型已保存: {model_path} ({Path(model_path).stat().st_size / 1024:.0f} KB)")

    # 加载
    with open(model_path, 'r', encoding='utf-8') as f:
        model_loaded = model_from_json(json.load(f))
    logger.info("模型加载成功")

    # 验证：相同输入应该生成相同预测
    future = model_loaded.make_future_dataframe(periods=30)
    forecast = model_loaded.predict(future)
    logger.info(f"加载后预测: 未来30天均值 = {forecast['yhat'].tail(30).mean():.1f}")

    return model_loaded

model = demo_serialization()

# ============================================================
# 2. ProphetPipeline — 生产管道类
# ============================================================
class ProphetPipeline:
    """生产级 Prophet 预测管道"""

    def __init__(self, model_params=None, model_path=None):
        self.model_params = model_params or {
            'yearly_seasonality': True,
            'weekly_seasonality': True,
        }
        self.model_path = model_path
        self.model = None
        self.last_train_time = None
        self.forecast_log = []

    def extract_data(self, data_source):
        """数据提取（模拟从数据库查询）"""
        if isinstance(data_source, pd.DataFrame):
            df = data_source.copy()
        else:
            raise NotImplementedError("接入你的数据库连接")
        df['ds'] = pd.to_datetime(df['ds'])
        return df[['ds', 'y']].sort_values('ds')

    def check_quality(self, df):
        """数据质量前置检查"""
        checks = {
            '最少30行': len(df) >= 30,
            'y 不是全 NaN': df['y'].notna().any(),
            'y 有方差': df['y'].std() > 1e-6,
            'ds 单调递增': df['ds'].is_monotonic_increasing,
        }
        failed = [k for k, v in checks.items() if not v]
        if failed:
            raise ValueError(f"质量检查失败: {failed}")
        logger.info(f"质量检查通过 ({len(df)} 行)")

    def fit_or_load(self, df):
        """训练新模型或加载已有模型"""
        if self.model_path and Path(self.model_path).exists():
            try:
                with open(self.model_path, 'r', encoding='utf-8') as f:
                    self.model = model_from_json(json.load(f))
                logger.info(f"从 {self.model_path} 加载模型")
                return
            except Exception as e:
                logger.warning(f"模型加载失败 ({e})，重新训练")

        logger.info("开始训练新模型...")
        self.model = Prophet(**self.model_params)
        self.model.fit(df)
        self.last_train_time = datetime.now()
        logger.info("模型训练完成")

    def predict(self, periods=90, freq='D'):
        """生成预测"""
        if self.model is None:
            raise RuntimeError("模型未加载，先调用 fit_or_load")
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)
        return forecast

    def check_reasonableness(self, forecast, historical, periods=90):
        """合理性检查"""
        recent_mean = historical['y'].tail(30).mean()
        forecast_mean = forecast['yhat'].tail(periods).mean()

        ratio = forecast_mean / recent_mean if recent_mean > 0 else 1
        warnings = []

        if ratio > 2 or ratio < 0.5:
            warnings.append(f"预测均值 ({forecast_mean:.0f}) 偏离近期 ({recent_mean:.0f}), ratio={ratio:.1f}")

        if historical['y'].min() >= 0 and (forecast['yhat'] < 0).any():
            warnings.append("预测含负值")

        for w in warnings:
            logger.warning(f"合理性检查: {w}")

        return len(warnings) == 0

    def save_model(self):
        """持久化模型"""
        if self.model and self.model_path:
            with open(self.model_path, 'w', encoding='utf-8') as f:
                json.dump(model_to_json(self.model), f, ensure_ascii=False)
            logger.info(f"模型已保存: {self.model_path}")

    def run(self, data_source, periods=90):
        """完整管道"""
        # 1 提取
        df = self.extract_data(data_source)

        # 2 质量检查
        self.check_quality(df)

        # 3 训练/加载
        self.fit_or_load(df)

        # 4 预测
        forecast = self.predict(periods)

        # 5 合理性检查
        self.check_reasonableness(forecast, df, periods)

        # 6 保存
        self.save_model()

        # 7 记录
        self.forecast_log.append({
            'timestamp': datetime.now(),
            'periods': periods,
            'forecast_mean': forecast['yhat'].tail(periods).mean(),
        })

        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)

# ============================================================
# 3. 运行管道
# ============================================================
pipeline = ProphetPipeline(
    model_params={
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'changepoint_prior_scale': 0.05,
    },
    model_path='models/production_model.json',
)

forecast = pipeline.run(df_production, periods=30)

print("\n" + "=" * 50)
print("生产管道输出 — 未来30天预测")
print("=" * 50)
print(forecast.to_string())

# ============================================================
# 4. 预测监控
# ============================================================
class ForecastMonitor:
    """预测效果监控器"""

    def __init__(self):
        self.log = []

    def record(self, ds, yhat, yhat_lower, yhat_upper, actual=None):
        self.log.append({
            'ds': ds, 'yhat': yhat,
            'yhat_lower': yhat_lower, 'yhat_upper': yhat_upper,
            'actual': actual,
        })

    def check(self, window=7):
        """检查最近 N 天的表现"""
        df = pd.DataFrame(self.log[-window:])
        df = df.dropna(subset=['actual'])

        if len(df) < 3:
            return {'mape': None, 'coverage': None, 'alerts': ['数据不足']}

        df['error'] = df['actual'] - df['yhat']
        df['ape'] = np.abs(df['error'] / df['actual']) * 100

        mape = df['ape'].mean()
        coverage = ((df['actual'] >= df['yhat_lower']) &
                    (df['actual'] <= df['yhat_upper'])).mean()

        alerts = []
        if mape > 25:
            alerts.append(f'MAPE={mape:.1f}% (高于25%)')
        if coverage < 0.5:
            alerts.append(f'覆盖率={coverage:.1%} (低于50%)')

        return {'mape': mape, 'coverage': coverage, 'alerts': alerts}

# 模拟监控
monitor = ForecastMonitor()
for i in range(10):
    ds = dates[-10 + i]
    fc_idx = forecast['ds'] == ds
    actual = df_production.loc[df_production['ds'] == ds, 'y'].values
    monitor.record(
        ds=ds,
        yhat=forecast.loc[fc_idx, 'yhat'].values[0] if fc_idx.any() else np.nan,
        yhat_lower=forecast.loc[fc_idx, 'yhat_lower'].values[0] if fc_idx.any() else np.nan,
        yhat_upper=forecast.loc[fc_idx, 'yhat_upper'].values[0] if fc_idx.any() else np.nan,
        actual=actual[0] if len(actual) > 0 else None,
    )

status = monitor.check()
print("\n" + "=" * 50)
print("监控报告")
print("=" * 50)
print(f"  MAPE:     {status['mape']:.1f}%" if status['mape'] else "  MAPE:     N/A")
print(f"  区间覆盖率: {status['coverage']:.1%}" if status['coverage'] else "  区间覆盖率: N/A")
for alert in status['alerts']:
    print(f"  {alert}")

print("\n完成！")
