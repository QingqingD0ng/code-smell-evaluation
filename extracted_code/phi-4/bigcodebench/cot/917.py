import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from statsmodels.tsa.arima.model import ARIMA

def task_func(df: pd.DataFrame) -> tuple:
    df = df.set_index('date')
    model = ARIMA(df['closing_price'], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=7).tolist()
    
    fig, ax = plt.subplots()
    df['closing_price'].plot(ax=ax, label='Observed')
    ax.plot(pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=7), forecast, label='Forecast', color='red')
    ax.legend()
    
    return forecast, ax