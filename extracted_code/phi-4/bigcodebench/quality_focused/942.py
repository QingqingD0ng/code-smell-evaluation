import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

START_DATE = '2016-01-01'
PERIODS = 13
FREQ = 'WOM-2FRI'
CATEGORIES = ['Electronics', 'Fashion', 'Home & Kitchen', 'Automotive', 'Sports']

def task_func(start_date=START_DATE, periods=PERIODS, freq=FREQ, categories=CATEGORIES):
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    data = []

    for date in dates:
        for category in categories:
            sales = np.random.randint(100, 500)
            data.append([date, category, sales])

    df = pd.DataFrame(data, columns=['Date', 'Category', 'Sales'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for category in categories:
        category_data = df[df['Category'] == category]
        ax.plot(category_data['Date'], category_data['Sales'], marker='o', label=category)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.set_title('Sales Report')
    ax.legend()
    ax.grid(True)
    
    return df, ax