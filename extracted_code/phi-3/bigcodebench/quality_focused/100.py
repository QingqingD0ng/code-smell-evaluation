import matplotlib.pyplot as plt
import pandas as pd
import random
from datetime import datetime, timedelta

def task_func(seed=42):
    random.seed(seed)
    start_date = datetime.now() - timedelta(days=30)
    dates = [start_date + timedelta(days=i) for i in range(30)]
    values = [random.random() for _ in range(30)]
    data = pd.DataFrame({'Date': dates, 'Value': values})
    fig, ax = plt.subplots()
    ax.plot(data['Date'], data['Value'], label='Random Time Series Data')
    ax.set_title('Random Time Series Data')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.tick_params(axis='both', which='major', labelsize=10, labelrotation=45)
    ax.xaxis.set_major_formatter(plt.DateFormatter('%Y-%m-%d'))
    fig.legend()
    return ax