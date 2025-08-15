import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

def task_func(df, freq='D', decomposition_model='multiplicative'):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input df must be a pandas DataFrame.")
    required_columns = {'group', 'date', 'value'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        raise ValueError("DataFrame 'date' column must be of datetime type.")
    if df['value'].dtype not in [float, int]:
        raise ValueError("DataFrame 'value' column must contain numeric data.")
    if decomposition_model not in ['additive','multiplicative']:
        raise ValueError("decomposition_model must be either 'additive' or'multiplicative'.")
    if freq not in ['D', 'W', 'M', 'A']:
        raise ValueError("freq must be one of 'D' (daily), 'W' (weekly), 'M' (monthly), 'A' (yearly).")

    # Ensure the data is sorted by date
    df.sort_values('date', inplace=True)

    # Decompose the time series
    decomposition = seasonal_decompose(df['value'], model=decomposition_model, period=freq)
    
    # Plot the decomposition
    plt.figure(figsize=(14, 8))
    ax = plt.subplot(311)
    plt.plot(df['date'], df['value'], label='Original')
    plt.title('Time Series Decomposition')
    plt.ylabel('Value')
    plt.legend()
    
    ax = plt.subplot(312)
    plt.plot(decomposition.trend, label='Trend')
    plt.title('Trend')
    plt.ylabel('Trend')
    plt.legend()

    ax