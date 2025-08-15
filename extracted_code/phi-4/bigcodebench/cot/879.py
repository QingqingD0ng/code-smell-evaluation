import pandas as pd
from scipy.stats import chi2_contingency

def task_func(data, col1, col2):
    if data.empty:
        raise ValueError("Data is empty.")
    if col1 not in data.columns or col2 not in data.columns:
        raise ValueError(f"Columns {col1} or {col2} are not in data.")
    col1_data = data[col1]
    col2_data = data[col2]
    if not pd.api.types.is_categorical_dtype(col1_data) and not pd.api.types.is_object_dtype(col1_data):
        raise TypeError(f"Column {col1} contains non-categorical data.")
    if not pd.api.types.is_categorical_dtype(col2_data) and not pd.api.types.is_object_dtype(col2_data):
        raise TypeError(f"Column {col2} contains non-categorical data.")
    contingency_table = pd.crosstab(col1_data, col2_data)
    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        raise ValueError("One or both columns do not have multiple categories.")
    if (contingency_table < 5).any().any():
        raise ValueError("Some categories have less than 5 observations.")
    _, p_value, _, _ = chi2_contingency(contingency_table)
    return p_value