import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

def task_func(df, freq='D', decomposition_model='multiplicative'):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    required_columns = {'group', 'date', 'value'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}.")
    
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        raise ValueError("The 'date' column must be of datetime type.")
    
    if not pd.api.types.is_numeric_dtype(df['value']):
        raise ValueError("The 'value' column must contain numeric data.")
    
    if decomposition_model not in ['additive','multiplicative']:
        raise ValueError("Decomposition model must be 'additive' or'multiplicative'.")
    
    df = df.set_index('date')
    decomposition = seasonal_decompose(df['value'], model=decomposition_model, period=freq)
    
    fig, ax = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    decomposition.plot(ax=ax)
    ax[0].set_title('Time Series Decomposition')
    ax[-1].set_xlabel('Date')
    ax[-1].set_ylabel('Value')
    
    return decomposition, ax