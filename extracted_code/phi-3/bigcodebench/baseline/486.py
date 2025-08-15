import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def task_func(start_time, end_time, step, trend, seed=42):
    np.random.seed(seed)
    datetime_start = datetime.datetime.utcfromtimestamp(start_time / 1000)
    datetime_end = datetime.datetime.utcfromtimestamp(end_time / 1000)
    time_index = pd.date_range(datetime_start, datetime_end, freq=f'{step}S')
    values = np.random.randn(len(time_index)) * 100 + trend * np.arange(len(time_index))
    df = pd.DataFrame({'Time': time_index, 'Value': values})
    df.set_index('Time', inplace=True)
    fig, ax = plt.subplots()
    ax.plot(df.index, df['Value'])
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    return ax