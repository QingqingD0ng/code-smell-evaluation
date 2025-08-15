import matplotlib.pyplot as plt
import pandas as pd
import random
from datetime import datetime, timedelta

def task_func(seed=42):
    random.seed(seed)
    dates = [datetime.now() - timedelta(days=i) for i in range(30)]
    values = [random.uniform(0, 100) for _ in range(30)]
    
    plt.style.use('seaborn-whitegrid')
    plt.rcParams['font.family'] = 'Arial'
    
    df = pd.DataFrame({'Date': dates, 'Value': values})
    df.sort_values('Date', inplace=True)
    
    ax = df.plot(x='Date', y='Value', kind='line', title='Random Time Series Data', legend=False)
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    
    return ax