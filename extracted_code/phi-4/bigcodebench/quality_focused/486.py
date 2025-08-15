import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def task_func(start_time, end_time, step, trend, seed=42):
    np.random.seed(seed)
    
    timestamps = pd.date_range(
        start=datetime.utcfromtimestamp(start_time / 1000),
        end=datetime.utcfromtimestamp(end_time / 1000),
        freq=f'{step}ms'
    )
    
    values = np.random.normal(size=len(timestamps)) + trend * np.arange(len(timestamps))
    
    df = pd.DataFrame({'Time': timestamps, 'Value': values})
    
    fig, ax = plt.subplots()
    ax.plot(df['Time'], df['Value'])
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return ax