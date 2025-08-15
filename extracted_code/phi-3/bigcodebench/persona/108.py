import pandas as pd

import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose


def task_func(df, freq='D', decomposition_model='multiplicative'):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame.")
    required_columns = {'group', 'date', 'value'}
    if not required_columns.issubset(df.columns):
        raise ValueError("DataFrame must include 'group', 'date', and 'value' columns.")
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        raise ValueError("'date' column must be datetime type.")
    if decomposition_model not in ['additive','multiplicative']:
        raise ValueError("decomposition_model must be 'additive' or'multiplicative'.")
    
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    
    decomposition = seasonal_decompose(df['value'], model=decomposition_model, period=pd.infer_freq(df.index))
    ax = plt.gca()
    ax.set_title('Time Series Decomposition')
    ax.set_ylabel('Value')
    
    return decomposition, ax


# Example usage:

# df = pd.DataFrame({
#     "group": ["A"] * 14,
#     "date": pd.to_datetime(["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04",
#                            "2022-01-05", "2022-01-06", "2022-01-07", "2022-01-08",
#                            "2022-01-09", "2022-01