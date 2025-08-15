import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def task_func(data, col1, col2):
    if data.empty:
        raise ValueError("Data is empty.")
    if col1 not in data or col2 not in data:
        raise ValueError(f"Column names {col1} or {col2} not in data.")
    if data[col1].dtype not in [object, 'category'] or data[col2].dtype not in [object, 'category']:
        raise TypeError("Columns must be categorical.")
    if data[col1].nunique() < 2 or data[col2].nunique() < 2:
        raise ValueError("One or both columns do not have multiple categories.")
    
    contingency_table = pd.crosstab(data[col1], data[col2])
    if (contingency_table < 5).any().any():
        raise ValueError("Some categories have less than 5 observations.")
    
    _, p_value, _, _ = chi2_contingency(contingency_table)
    return p_value