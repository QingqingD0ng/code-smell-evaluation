from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def task_func(start_time, end_time, step, trend, seed=42):
    np.random.seed(seed)

    timestamps = pd.date_range(start=datetime.utcfromtimestamp(start_time / 1000),
                               end=datetime.utcfromtimestamp(end_time / 1000),
                               freq=f'{step}ms')

    values = np.random.normal(size=len(timestamps))
    values += trend * np.arange(len(timestamps))

    df = pd.DataFrame({'Time': timestamps, 'Value': values})

    ax = df.plot(x='Time', y='Value', kind='line', legend=False)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')

    return ax

# Example usage:
# ax = task_func(0, 10000, 100, 0.001)
# plt.show()