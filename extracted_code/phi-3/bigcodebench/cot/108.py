import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

def task_func(df, freq='D', decomposition_model='multiplicative'):
    required_columns = {'group', 'date', 'value'}
    if not isinstance(df, pd.DataFrame) or not required_columns.issubset(df.columns):
        raise ValueError("Input 'df' must be a DataFrame with columns 'group', 'date', and 'value'.")
    
    if freq not in ['D', 'W', 'M', 'A', 'Q', 'Y']:
        raise ValueError("Invalid frequency string. Valid options are 'D', 'W', 'M', 'A', 'Q', 'Y'.")
    
    if decomposition_model not in ['additive','multiplicative']:
        raise ValueError("Decomposition model must be 'additive' or'multiplicative'.")

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    result = seasonal_decompose(df.pivot(index='date', columns='group', values='value'), model=decomposition_model, freq=freq)
    
    fig, ax = plt.subplots()
    result.plot(ax=ax)
    ax.set_title('Time Series Decomposition')
    ax.set_ylabel('Value')
    plt.show()
    
    return result, ax