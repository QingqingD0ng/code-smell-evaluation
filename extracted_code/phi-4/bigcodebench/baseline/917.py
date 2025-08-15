import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from statsmodels.tsa.arima.model import ARIMA
from typing import List, Tuple

def task_func(df: pd.DataFrame) -> Tuple[List[float], Axes]:
    df.set_index('date', inplace=True)
    model = ARIMA(df['closing_price'], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=7)
    fig, ax = plt.subplots()
    df['closing_price'].plot(ax=ax, label='Observed')
    forecast.plot(ax=ax, label='Forecast', style='--')
    ax.legend()
    return forecast.tolist(), ax