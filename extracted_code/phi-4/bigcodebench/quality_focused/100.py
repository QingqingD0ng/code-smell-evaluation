import matplotlib.pyplot as plt
import pandas as pd
import random
from datetime import datetime, timedelta

def task_func(seed=42):
    random.seed(seed)

    # Generate dates for the past 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=29)
    dates = [start_date + timedelta(days=i) for i in range(30)]

    # Generate random values
    values = [random.random() for _ in range(30)]

    # Create DataFrame
    data = pd.DataFrame({'Date': dates, 'Value': values})
    data.set_index('Date', inplace=True)

    # Plot data
    plt.style.use('seaborn-whitegrid')
    ax = data.plot(kind='line', linewidth=2)

    # Set plot labels and title
    ax.set_xlabel('Date', fontname='Arial')
    ax.set_ylabel('Value', fontname='Arial')
    ax.set_title('Random Time Series Data', fontname='Arial')

    return ax