import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Constants
START_DATE = '2016-01-01'
PERIODS = 13
FREQ = 'WOM-2FRI'
CATEGORIES = ['Electronics', 'Fashion', 'Home & Kitchen', 'Automotive', 'Sports']

def task_func(start_date=START_DATE, periods=PERIODS, freq=FREQ, categories=CATEGORIES):
    # Generate date range
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    
    # Create DataFrame
    data = {'Date': [], 'Category': [], 'Sales': []}
    for date in dates:
        for category in categories:
            sales = np.random.randint(100, 500)
            data['Date'].append(date)
            data['Category'].append(category)
            data['Sales'].append(sales)
    
    df = pd.DataFrame(data)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    for category in categories:
        category_data = df[df['Category'] == category]
        ax.plot(category_data['Date'], category_data['Sales'], marker='o', label=category)
    
    ax.set_title('Sales Report')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.legend()
    ax.grid(True)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return df, ax