from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def task_func(start_time, end_time, step, trend, seed=42):
    if end_time <= start_time:
        raise ValueError("end_time must be greater than start_time")
    if step <= 0:
        raise ValueError("step must be greater than 0")
    
    np.random.seed(seed)
    
    times = pd.date_range(
        start=datetime.fromtimestamp(start_time / 1000),
        end=datetime.fromtimestamp(end_time / 1000),
        freq=f'{step}L'
    )
    
    values = np.random.normal(size=len(times))
    values += trend * np.arange(len(times))
    
    df = pd.DataFrame({'Time': times, 'Value': values})
    
    fig, ax = plt.subplots()
    ax.plot(df['Time'], df['Value'])
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    
    return ax