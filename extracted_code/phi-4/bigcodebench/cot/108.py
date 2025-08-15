import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

def task_func(df, freq='D', decomposition_model='multiplicative'):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    required_columns = {'group', 'date', 'value'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")
    
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        raise ValueError("Column 'date' must be datetime type.")
    
    if not pd.api.types.is_numeric_dtype(df['value']):
        raise ValueError("Column 'value' must be numeric type.")
    
    if not isinstance(freq, str):
        raise ValueError("Frequency must be a string.")
    
    if decomposition_model not in ['additive','multiplicative']:
        raise ValueError("Decomposition model must be 'additive' or'multiplicative'.")
    
    df.set_index('date', inplace=True)
    
    decomposition = seasonal_decompose(df['value'], model=decomposition_model, period=freq)
    
    fig, ax = plt.subplots()
    decomposition.plot()
    ax.set_title('Time Series Decomposition')
    ax.set_ylabel('Value')
    
    return decomposition, ax