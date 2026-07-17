"""
第 13 章：实战案例
==================
四个完整场景的端到端演示：
- 案例 1: 零售日销售
- 案例 2: 网站日流量
- 案例 3: 餐厅日营收
- 案例 4: 能源日负荷

每个案例都是独立的函数，可以直接替换为自己的数据。
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import matplotlib.pyplot as plt

# ============================================================
# 工具函数
# ============================================================
def generate_casestudy_data(case='retail'):
    """为不同案例生成模拟数据"""
    np.random.seed(42)

    if case == 'retail':
        dates = pd.date_range('2022-01-01', '2024-12-31', freq='D')
        n = len(dates)
        trend = np.linspace(10000, 18000, n)
        weekly = 0.25 * np.sin(2 * np.pi * np.arange(n) / 7)
        yearly = 0.1 * np.sin(2 * np.pi * np.arange(n) / 365.25)
        noise = np.random.randn(n) * 0.03

        y = trend * (1 + weekly + yearly + noise)

        # 大促日
        for year in [2022, 2023]:
            for month_day in [('06-15', '06-20'), ('11-09', '11-13')]:
                start, end = f'{year}-{month_day[0]}', f'{year}-{month_day[1]}'
                mask = (dates >= start) & (dates <= end)
                y[mask] *= 1.5

        # 周日关门
        y[dates.dayofweek == 6] *= 0.7
        return pd.DataFrame({'ds': dates, 'y': np.maximum(y, 100)})

    elif case == 'web_traffic':
        dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
        n = len(dates)
        trend = np.linspace(5000, 7000, n)
        weekly = 0.15 * np.sin(2 * np.pi * np.arange(n) / 7 - 1)  # 工作日高
        yearly = 0.05 * np.sin(2 * np.pi * np.arange(n) / 365.25)
        noise = np.random.randn(n) * 100

        y = trend * (1 + weekly + yearly) + noise

        # 节假日暴跌
        for holiday_range in [('2023-10-01', '2023-10-07'), ('2024-02-10', '2024-02-16')]:
            mask = (dates >= holiday_range[0]) & (dates <= holiday_range[1])
            y[mask] *= 0.4

        return pd.DataFrame({'ds': dates, 'y': np.maximum(y, 100)})

    elif case == 'energy':
        dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
        n = len(dates)
        trend = np.linspace(40000, 55000, n)
        temperature = 15 + 12 * np.sin(2 * np.pi * np.arange(n) / 365.25 - np.pi/2) \
                      + np.random.randn(n) * 3

        # HDD 和 CDD
        hdd = np.maximum(18 - temperature, 0)
        cdd = np.maximum(temperature - 22, 0)
        weekly = 0.05 * np.sin(2 * np.pi * np.arange(n) / 7 - 0.5)
        noise = np.random.randn(n) * 500

        y = trend * (1 + weekly) + hdd * 500 + cdd * 800 + noise

        df = pd.DataFrame({
            'ds': dates,
            'y': np.maximum(y, 10000),
            'temperature': temperature,
            'hdd': hdd,
            'cdd': cdd,
        })
        return df

    else:
        raise ValueError(f"Unknown case: {case}")


# ============================================================
# 案例 1: 零售日销售
# ============================================================
def case_retail():
    print("\n" + "=" * 60)
    print("案例 1: 零售日销售预测")
    print("=" * 60)

    df = generate_casestudy_data('retail')

    # 自定义大促日
    promo_events = pd.DataFrame({
        'holiday': ['618', '618', '双11', '双11', '618', '618', '双11', '双11'],
        'ds': pd.to_datetime([
            '2022-06-18', '2023-06-18', '2022-11-11', '2023-11-11',
            '2024-06-18', '2025-06-18', '2024-11-11', '2025-11-11',
        ]),
        'lower_window': -3,
        'upper_window': 2,
    })

    model = Prophet(
        seasonality_mode='multiplicative',
        yearly_seasonality=15,
        weekly_seasonality=5,
        changepoint_prior_scale=0.1,
        holidays_prior_scale=15.0,
        holidays=promo_events,
    )
    model.add_country_holidays(country_name='CN')
    model.fit(df)

    future = model.make_future_dataframe(90)
    forecast = model.predict(future)

    print(f"  未来90天日均预测: {forecast['yhat'].tail(90).mean():.0f}")
    print(f"  预测范围: {forecast['yhat'].tail(90).min():.0f} ~ {forecast['yhat'].tail(90).max():.0f}")

    model.plot_components(forecast)
    plt.suptitle('案例1: 零售日销售 — 成分分解')
    plt.tight_layout()
    plt.show()

    return model, forecast


# ============================================================
# 案例 2: 网站日流量
# ============================================================
def case_web_traffic():
    print("\n" + "=" * 60)
    print("案例 2: 网站日流量预测")
    print("=" * 60)

    df = generate_casestudy_data('web_traffic')

    # 节假日作为负效应
    holidays_web = pd.DataFrame({
        'holiday': ['国庆假期', '春节假期', '五一假期'],
        'ds': pd.to_datetime(['2023-10-01', '2024-02-10', '2024-05-01']),
        'upper_window': [6, 6, 4],
    })

    model = Prophet(
        seasonality_mode='additive',
        weekly_seasonality=7,
        yearly_seasonality=10,
        changepoint_prior_scale=0.1,
        holidays_prior_scale=20.0,
        holidays=holidays_web,
    )
    model.fit(df)

    future = model.make_future_dataframe(30)
    forecast = model.predict(future)

    print(f"  未来30天日均预测: {forecast['yhat'].tail(30).mean():.0f}")
    print(f"  节假日总效应: {forecast['holidays'].min():.0f} ~ {forecast['holidays'].max():.0f}")

    fig = model.plot_components(forecast)
    plt.suptitle('案例2: 网站日流量 — 成分分解')
    plt.tight_layout()
    plt.show()

    return model, forecast


# ============================================================
# 案例 3: 餐厅日营收（多门店 + 额外回归量）
# ============================================================
def case_restaurant():
    print("\n" + "=" * 60)
    print("案例 3: 餐厅日营收预测（含额外回归量）")
    print("=" * 60)

    np.random.seed(42)
    dates = pd.date_range('2022-01-01', '2024-12-31', freq='D')
    n = len(dates)

    trend = np.linspace(2000, 3500, n)
    weekly = 300 * np.sin(2 * np.pi * np.arange(n) / 7)
    yearly = 400 * np.sin(2 * np.pi * np.arange(n) / 365.25)

    # 外部变量
    promo = np.where(np.arange(n) % 30 < 3, 1, 0)
    temperature = 15 + 10 * np.sin(2 * np.pi * np.arange(n) / 365.25)
    temp_deviation = np.abs(temperature - 25)
    noise = np.random.randn(n) * 150

    df = pd.DataFrame({
        'ds': dates,
        'y': trend + weekly + yearly + promo * 2000 + noise - temp_deviation * 30,
        'promo': promo,
        'temperature': temperature,
        'temp_deviation': temp_deviation,
    })

    # 划分训练/测试
    train = df.iloc[:-90]
    test = df.iloc[-90:]

    model = Prophet(
        yearly_seasonality=15,
        weekly_seasonality=5,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.1,
    )
    model.add_regressor('promo')
    model.add_regressor('temperature')
    model.add_regressor('temp_deviation')
    model.fit(train)

    future = model.make_future_dataframe(90)
    future = future.merge(test[['ds', 'promo', 'temperature', 'temp_deviation']],
                          on='ds', how='left')
    future.fillna({'promo': 0, 'temperature': 20, 'temp_deviation': 5}, inplace=True)
    forecast = model.predict(future)

    # 评估
    comparison = test[['ds', 'y']].merge(forecast[['ds', 'yhat']], on='ds')
    mae = (comparison['y'] - comparison['yhat']).abs().mean()
    print(f"  测试集 MAE: {mae:.0f}")
    print(f"  回归量平均效应: promo={forecast['promo'].mean():.0f}, "
          f"temperature={forecast['temperature'].mean():.0f}")

    return model, forecast


# ============================================================
# 案例 4: 能源日负荷
# ============================================================
def case_energy():
    print("\n" + "=" * 60)
    print("案例 4: 能源日负荷预测")
    print("=" * 60)

    df = generate_casestudy_data('energy')

    model = Prophet(
        seasonality_mode='multiplicative',
        yearly_seasonality=20,
        weekly_seasonality=5,
        changepoint_prior_scale=0.01,
        seasonality_prior_scale=15.0,
        interval_width=0.90,
    )
    model.add_regressor('hdd')
    model.add_regressor('cdd')
    model.fit(df)

    future = model.make_future_dataframe(14)

    # 未来温度需要外部提供（这里用历史均值模拟）
    future['hdd'] = df['hdd'].tail(365).mean()
    future['cdd'] = df['cdd'].tail(365).mean()
    forecast = model.predict(future)

    print(f"  未来14天日均负荷: {forecast['yhat'].tail(14).mean():.0f} MWh")
    print(f"  90%区间: [{forecast['yhat_lower'].tail(14).mean():.0f}, "
          f"{forecast['yhat_upper'].tail(14).mean():.0f}]")
    print(f"  HDD平均效应: {forecast['hdd'].mean():.0f}")
    print(f"  CDD平均效应: {forecast['cdd'].mean():.0f}")

    fig = model.plot_components(forecast)
    plt.suptitle('案例4: 能源日负荷 — 成分分解')
    plt.tight_layout()
    plt.show()

    return model, forecast


# ============================================================
# 运行全部案例
# ============================================================
if __name__ == '__main__':
    print("Prophet 实战案例 — 四场景演示")

    case_retail()
    case_web_traffic()
    case_restaurant()
    case_energy()

    print("\n" + "=" * 60)
    print("全部 4 个案例运行完成！")
    print("=" * 60)
