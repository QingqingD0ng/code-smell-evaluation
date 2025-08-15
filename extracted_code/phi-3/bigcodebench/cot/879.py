import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def task_func(data, col1, col2):
    if data.empty:
        raise ValueError("DataFrame is empty")
    if col1 not in data.columns or col2 not in data.columns:
        raise ValueError("Column names must be in DataFrame")
    
    col1_data = data[col1].astype('category')
    col2_data = data[col2].astype('category')
    
    if col1_data.dtypes!= 'category' or col2_data.dtypes!= 'category':
        raise TypeError("Columns must contain categorical data")
    
    table = pd.crosstab(col1_data, col2_data)
    
    if table.isnull().values.any():
        raise ValueError("Some categories have less than 5 observations")
    
    chi2, p, dof, expected = chi2_contingency(table)
    return p