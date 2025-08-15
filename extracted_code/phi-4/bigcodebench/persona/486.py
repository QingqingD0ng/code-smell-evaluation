import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def task_func(start_time, end_time, step, trend, seed=42):
    np.random.seed(seed)
    
    # Create a date range from start_time to end_time with the given step
    time_range = pd.date_range(start=datetime.fromtimestamp(start_time / 1000.0),
                               end=datetime.fromtimestamp(end_time / 1000.0),
                               freq=f'{step}ms')
    
    # Generate random values from a normal distribution
    values = np.random.randn(len(time_range))
    
    # Add a linear trend to the values
    trend_values = trend * np.arange(len(time_range))
    values_with_trend = values + trend_values
    
    # Create a DataFrame
    df = pd.DataFrame({'Time': time_range, 'Value': values_with_trend})
    
    # Plot the time series
    fig, ax = plt.subplots()
    ax.plot(df['Time'], df['Value'])
    
    # Set labels
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    
    return ax