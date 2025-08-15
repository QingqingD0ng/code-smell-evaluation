import pandas as pd
import numpy as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from typing import List, Tuple

def task_func(df: pd.DataFrame) -> Tuple[List[float], plt.Axes]:
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    model = ARIMA(df['closing_price'], order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=7)[0]
    fig, ax = plt.subplots()
    ax.plot(df['closing_price'], label='Historical')
    ax.plot(pd.date_range(df.index[-1], periods=7, freq='D'), forecast, label='Forecast')
    ax.legend()
    return forecast, ax