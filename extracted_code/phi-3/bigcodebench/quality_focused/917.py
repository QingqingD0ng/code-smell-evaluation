import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

def task_func(df: pd.DataFrame) -> Tuple[List[float], Axes]:
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    model = ARIMA(df['closing_price'], order=(1,1,1))
    model_fit = model.fit()
    forecast, stderr, conf_int = model_fit.forecast(7)
    fig, ax = plt.subplots()
    ax.plot(df['closing_price'], label='Historical')
    ax.plot(pd.date_range(start=df.index[-1], periods=8, freq='D'), forecast, label='Forecast')
    ax.fill_between(pd.date_range(start=df.index[-1], periods=8, freq='D'), conf_int[:,0], conf_int[:,1], color='pink', alpha=0.3)
    ax.legend(loc='upper left')
    return forecast.tolist(), ax