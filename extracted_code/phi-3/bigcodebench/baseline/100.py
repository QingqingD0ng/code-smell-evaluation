import matplotlib.pyplot as plt
import pandas as pd
import random
from datetime import datetime, timedelta

def task_func(seed=42):
    random.seed(seed)
    dates = pd.date_range(end=datetime.today(), periods=30)
    values = [random.uniform(100, 200) for _ in range(30)]
    
    df = pd.DataFrame({'Date': dates, 'Value': values})
    ax = df.plot(kind='line', fontsize='12')
    
    ax.set_xlabel('Date', fontdict={'family': 'Arial'})
    ax.set_ylabel('Value', fontdict={'family': 'Arial'})
    ax.set_title('Random Time Series Data', fontdict={'family': 'Arial'})
    
    return ax