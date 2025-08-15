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
            data.append({'Date': date, 'Category': category, 'Sales': sales})

    df = pd.DataFrame(data)

    fig, ax = plt.subplots()
    for category in categories:
        category_data = df[df['Category'] == category]
        ax.plot(category_data['Date'], category_data['Sales'], marker='o', label=category)

    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.set_title('Sales Report by Category')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    return df, ax

# Example usage:
# df, ax = task_func(start_date='2020-01-01', periods=5, freq='W-MON', categories=['Electronics', 'Fashion'])
# df.to_string(index=False)