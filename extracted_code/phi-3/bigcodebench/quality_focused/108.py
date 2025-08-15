import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

def task_func(df, freq='D', decomposition_model='multiplicative'):
    # Validate input DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a DataFrame.")
    required_columns = {'group', 'date', 'value'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"df must contain the columns: {required_columns}")
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        raise ValueError("df['date'] must be a datetime type.")
    if not pd.api.types.is_numeric_dtype(df['value']):
        raise ValueError("df['value'] must be a numeric type.")
    
    # Validate frequency and decomposition model
    if freq not in ['D', 'W', 'M']:
        raise ValueError("freq must be 'D', 'W', or 'M'.")
    if decomposition_model not in ['additive','multiplicative']:
        raise ValueError("decomposition_model must be 'additive' or'multiplicative'.")
    
    # Group by 'group' and 'date' and resample to ensure regular frequency
    df_grouped = df.groupby('group').resample(freq, on='date').agg({'value':'mean'}).reset_index()
    
    # Perform decomposition
    decomposition = seasonal_decompose(df_grouped['value'], model=decomposition_model, freq=freq)
    
    # Plot decomposition
    fig, ax = plt.subplots(figsize=(10, 6))
    decomposition.plot(ax=ax)
    ax.set_title('Time Series Decomposition')
    ax.set_ylabel('Value')
    
    return decomposition, ax

# Example usage:
# df = pd.DataFrame({
#     "group": ["A"] * 14,
#     "date": pd.to_datetime