import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def task_func(data, col1, col2):
    # Check if data is empty
    if data.empty:
        raise ValueError("DataFrame is empty.")
    
    # Check if columns are in DataFrame
    if col1 not in data.columns or col2 not in data.columns:
        raise ValueError("One or both columns are not in the DataFrame.")
    
    # Check if columns contain categorical data
    if not pd.api.types.is_categorical_dtype(data[col1]) or not pd.api.types.is_categorical_dtype(data[col2]):
        raise TypeError("Columns must contain categorical data.")
    
    # Check for sufficient category counts and observations
    if data[col1].nunique() < 2 or data[col2].nunique() < 2 or any(data.groupby(col1).size() < 5) or any(data.groupby(col2).size() < 5):
        raise ValueError("One or both columns do not have multiple categories or some categories have less than 5 observations.")
    
    # Create contingency table
    contingency_table = pd.crosstab(data[col1], data[col2])
    
    # Perform chi-square test of independence
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    return p