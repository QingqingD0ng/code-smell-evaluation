import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def task_func(start_date='2016-01-01', periods=13, freq='WOM-2FRI', categories=['Electronics', 'Fashion', 'Home & Kitchen', 'Automotive', 'Sports']):
    date_range = pd.date_range(start=start_date, periods=periods, freq=freq).to_frame(index=False, name='Date')
    sales_data = pd.DataFrame({
        'Date': np.tile(date_range['Date'], len(categories)),
        'Category': np.repeat(categories, periods),
        'Sales': np.random.randint(100, 500, size=(len(date_range) * len(categories)))
    })
    sales_data.index = [f'{row.Date.strftime("%Y-%m-%d")}, {row.Category}' for row in sales_data.itertuples()]
    ax = sales_data.groupby(['Date', 'Category'])['Sales'].sum().unstack().plot(kind='area', stacked=False)
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.set_title('Sales Report')
    plt.xticks(rotation=45)
    plt.legend(title='Category')
    return sales_data, ax