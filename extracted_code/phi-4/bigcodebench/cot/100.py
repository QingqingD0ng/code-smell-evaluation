import matplotlib.pyplot as plt
import pandas as pd
import random
from datetime import datetime, timedelta

def task_func(seed=42):
    random.seed(seed)
    
    # Generate date range for the past 30 days
    end_date = datetime.today()
    start_date = end_date - timedelta(days=29)
    date_range = pd.date_range(start_date, end_date)
    
    # Generate random data
    random_data = [random.uniform(0, 100) for _ in range(30)]
    
    # Create DataFrame
    df = pd.DataFrame({'Date': date_range, 'Value': random_data})
    
    # Plotting
    plt.style.use('default')
    plt.rcParams['font.family'] = 'Arial'
    plt.figure(figsize=(10, 6))
    ax = df.plot(x='Date', y='Value', kind='line', title='Random Time Series Data')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    
    return ax

# Example usage:
# ax = task_func()
# plt.show()