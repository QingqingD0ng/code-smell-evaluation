import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def task_func(data, col1, col2):
    if data.empty:
        raise ValueError("Input data is empty.")
    if col1 not in data.columns or col2 not in data.columns:
        raise ValueError("One or both columns are not in the DataFrame.")
    
    col1_data = data[col1]
    col2_data = data[col2]
    
    if not np.issubdtype(col1_data.dtype, np.object_) or not np.issubdtype(col2_data.dtype, np.object_):
        raise TypeError("One or both columns contain non-categorical data.")
    
    contingency_table = pd.crosstab(col1_data, col2_data)
    
    if (contingency_table < 5).any().any():
        raise ValueError("Some categories have less than 5 observations.")
    
    if len(contingency_table.index) < 2 or len(contingency_table.columns) < 2:
        raise ValueError("One or both columns do not have multiple categories.")
    
    chi2, p, _, _ = chi2_contingency(contingency_table)
    return p