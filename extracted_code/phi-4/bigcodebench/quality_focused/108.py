import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

def task_func(df, freq='D', decomposition_model='multiplicative'):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if not all(col in df.columns for col in ['group', 'date', 'value']):
        raise ValueError("DataFrame must contain 'group', 'date', and 'value' columns.")
    if df['date'].dtype!= 'datetime64[ns]':
        raise ValueError("'date' column must be of datetime type.")
    if df['value'].dtype not in [int, float]:
        raise ValueError("'value' column must contain numeric data.")
    if decomposition_model not in ['additive','multiplicative']:
        raise ValueError("decomposition_model must be 'additive' or'multiplicative'.")

    df = df.set_index('date').asfreq(freq)
    decomposition = seasonal_decompose(df['value'], model=decomposition_model, period=1)

    fig, ax = plt.subplots()
    decomposition.plot(ax=ax)
    ax.set_title('Time Series Decomposition')
    ax.set_ylabel('Value')
    plt.show()

    return decomposition, ax