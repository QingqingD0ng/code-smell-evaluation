import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

def task_func(df, freq='D', decomposition_model='multiplicative'):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if not {'group', 'date', 'value'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'group', 'date', and 'value' columns.")
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        raise ValueError("Column 'date' must be of datetime type.")
    if not pd.api.types.is_numeric_dtype(df['value']):
        raise ValueError("Column 'value' must be numeric.")
    if decomposition_model not in ['additive','multiplicative']:
        raise ValueError("decomposition_model must be 'additive' or'multiplicative'.")
    
    df.set_index('date', inplace=True)
    result = seasonal_decompose(df['value'], model=decomposition_model, period=1, extrapolate_trend='freq')
    
    fig, ax = plt.subplots()
    result.plot().suptitle('Time Series Decomposition', fontsize=16)
    ax.set_ylabel('Value')
    plt.show()
    
    return result, ax