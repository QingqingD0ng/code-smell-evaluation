import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def task_func(data, col1, col2):
    if data.empty:
        raise ValueError("Data is empty")
    if col1 not in data.columns or col2 not in data.columns:
        raise ValueError("One or both columns are not in data")
    if not pd.api.types.is_categorical_dtype(data[col1]) or not pd.api.types.is_categorical_dtype(data[col2]):
        raise TypeError("One or both columns contain non-categorical data")
    if data[col1].nunique() < 2 or data[col2].nunique() < 2:
        raise ValueError("One or both columns do not have multiple categories")
    
    contingency_table = pd.crosstab(data[col1], data[col2])
    if (contingency_table < 5).any().any():
        raise ValueError("Some categories have less than 5 observations")
    
    _, p_value, _, _ = chi2_contingency(contingency_table)
    return p_value