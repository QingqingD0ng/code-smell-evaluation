import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

def task_func(df, freq='D', decomposition_model='multiplicative', plot_title='Time Series Decomposition', ylabel='Value'):
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

    df_copy = df.copy()
    df_copy.set_index('date', inplace=True)
    
    # Determine period based on frequency
    if freq == 'D':
        period = 7  # Example: weekly seasonality for daily data
    elif freq == 'M':
        period = 12  # Example: yearly seasonality for monthly data
    else:
        period = 1  # Default to no seasonality if unknown

    result = seasonal_decompose(df_copy['value'], model=decomposition_model, period=period)

    fig, ax = plt.subplots()
    result.plot().suptitle(plot_title, fontsize=16)
    ax.set_ylabel(ylabel)
    plt.show()

    return result, ax