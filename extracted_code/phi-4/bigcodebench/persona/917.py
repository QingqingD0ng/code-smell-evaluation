import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from statsmodels.tsa.arima.model import ARIMA
from typing import List, Tuple

def task_func(df: pd.DataFrame) -> Tuple[List[float], Axes]:
    # Ensure the 'date' column is in datetime format and set it as the index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Fit the ARIMA model
    model = ARIMA(df['closing_price'], order=(5, 1, 0))
    model_fit = model.fit()

    # Forecast the next 7 days
    forecast, _, _ = model_fit.forecast(steps=7)

    # Plot the historical data and forecast
    fig, ax = plt.subplots()
    df['closing_price'].plot(ax=ax, label='Historical')
    forecast.plot(ax=ax, label='Forecast', linestyle='--')
    ax.legend()

    return forecast.tolist(), ax