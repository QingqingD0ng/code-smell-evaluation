import matplotlib.pyplot as plt
import pandas as pd
import random
from datetime import datetime, timedelta
from typing import Optional

def task_func(seed: Optional[int] = 42, days: int = 30, value_range: tuple = (0, 100)) -> plt.Axes:
    random.seed(seed)
    plt.style.use('seaborn-whitegrid')
    plt.rc('font', family='Arial')

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    date_range = pd.date_range(start_date, periods=days, freq='D')

    values = [random.uniform(*value_range) for _ in range(days)]
    data = pd.DataFrame({'Date': date_range, 'Value': values})

    ax = data.plot(x='Date', y='Value', kind='line', title='Random Time Series Data')
    ax.set(xlabel='Date', ylabel='Value')
    return ax