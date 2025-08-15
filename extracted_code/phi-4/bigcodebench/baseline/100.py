import matplotlib.pyplot as plt
import pandas as pd
import random
from datetime import datetime, timedelta

def task_func(seed=42):
    random.seed(seed)
    plt.style.use('seaborn-whitegrid')
    plt.rc('font', family='Arial')

    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    date_range = pd.date_range(start_date, periods=30, freq='D')

    values = [random.uniform(0, 100) for _ in range(30)]
    data = pd.DataFrame({'Date': date_range, 'Value': values})

    ax = data.plot(x='Date', y='Value', kind='line', title='Random Time Series Data')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    return ax