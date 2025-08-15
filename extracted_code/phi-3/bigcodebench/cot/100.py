import matplotlib.pyplot as plt
import pandas as pd
import random
from datetime import datetime, timedelta

def task_func(seed=42):
    random.seed(seed)
    start_date = datetime.now() - timedelta(days=30)
    dates = [start_date + timedelta(days=i) for i in range((datetime.now() - start_date).days + 1)]
    values = [random.random() * 100 for _ in range(len(dates))]
    data = pd.DataFrame({'Date': dates, 'Value': values})

    plt.style.use('seaborn-darkgrid')
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots()
    ax.plot(data['Date'], data['Value'])
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title('Random Time Series Data')
    return ax