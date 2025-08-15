import numpy as pd
import statsmodels.api as sm

def task_func(df: pd.DataFrame) -> Tuple[List[float], plt.Axes]:
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    
    model = sm.tsa.ARIMA(df['closing_price'], order=(1, 1, 1))
    model_fit = model.fit()
    
    forecast = model_fit.forecast(steps=7)[0]
    
    fig, ax = plt.subplots()
    ax.plot(df.index, df['closing_price'], label='Historical')
    ax.plot(pd.date_range(start=df.index[-1], periods=8)[1:], forecast, label='Forecast')
    ax.legend()
    
    return forecast, ax