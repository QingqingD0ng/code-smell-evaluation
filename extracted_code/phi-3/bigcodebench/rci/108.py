import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

def validate_inputs(df, freq, decomposition_model):
    # Validate DataFrame and columns
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input df must be a pandas DataFrame.")
    if not {'group', 'date', 'value'}.issubset(df.columns):
        raise ValueError("DataFrame must contain columns: 'group', 'date', 'value'")
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        raise ValueError("DataFrame 'date' column must be of datetime type.")
    if df['value'].dtype not in [float, int]:
        raise ValueError("DataFrame 'value' column must contain numeric data.")
    if decomposition_model not in ['additive','multiplicative']:
        raise ValueError("decomposition_model must be either 'additive' or'multiplicative'.")
    if freq not in ['D', 'W', 'M', 'A']:
        raise ValueError("freq must be one of 'D' (daily), 'W' (weekly), 'M' (monthly), 'A' (yearly).")

def plot_decomposition(decomposition, ax, decomposition_model):
    ax.set_title(f'{decomposition_model.capitalize()} Time Series Decomposition')
    ax.plot(decomposition.observed, label='Original')
    ax.plot(decomposition.trend, label='Trend')
    ax.plot(decomposition.seasonal, label='Seasonal')
    ax.plot(decomposition.resid, label='Residual')
    ax.set_ylabel('Value')
    ax.legend()

def task_func(df, freq='D', decomposition_model='multiplicative'):
    validate_inputs(df, freq, decomposition_model)
    df.sort_values('date', inplace=True)
    decomposition = seasonal_decompose(df['value'], model=decomposition