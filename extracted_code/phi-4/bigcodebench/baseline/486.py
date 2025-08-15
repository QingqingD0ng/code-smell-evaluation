from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def task_func(start_time, end_time, step, trend, seed=42):
    np.random.seed(seed)
    
    # Generate timestamps
    times = pd.date_range(start=datetime.fromtimestamp(start_time / 1000), 
                          end=datetime.fromtimestamp(end_time / 1000), 
                          freq=f'{step}ms')
    
    # Generate random values
    values = np.random.normal(size=len(times))
    
    # Add linear trend
    values += trend * np.arange(len(times))
    
    # Create DataFrame
    df = pd.DataFrame({'Time': times, 'Value': values})
    
    # Plotting
    fig, ax = plt.subplots()
    ax.plot(df['Time'], df['Value'])
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    
    return ax