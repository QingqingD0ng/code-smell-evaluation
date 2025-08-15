import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def task_func(start_date='2016-01-01', periods=13, freq='WOM-2FRI', categories=['Electronics', 'Fashion', 'Home & Kitchen', 'Automotive', 'Sports']):
    # Generate date range
    date_range = pd.date_range(start=start_date, periods=periods, freq=freq)
    
    # Random sales data generation
    np.random.seed(0)
    sales_data = {category: np.random.randint(low=50, high=500, size=len(date_range)) for category in categories}
    
    # Create DataFrame
    df = pd.DataFrame(sales_data, index=date_range)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Date'}, inplace=True)
    df['Sales'] = df.drop('Category', axis=1).sum(axis=1)
    
    # Plotting
    ax = df.plot(x='Date', y='Sales', kind='line', figsize=(10, 6), title='Sales Report by Category')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.legend(loc='upper left', labels=categories)
    
    return df, ax