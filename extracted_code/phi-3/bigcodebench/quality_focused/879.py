import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def task_func(data, col1, col2):
    if data.empty:
        raise ValueError("DataFrame is empty.")
    if col1 not in data.columns or col2 not in data.columns:
        raise ValueError(f"Column {col1} or {col2} not found in DataFrame.")
    if not pd.api.types.is_categorical_dtype(data[col1]) or not pd.api.types.is_categorical_dtype(data[col2]):
        raise TypeError("One or both columns contain non-categorical data.")
    if data[col1].nunique() < 2 or data[col2].nunique() < 2:
        raise ValueError("One or both columns do not have multiple categories.")
    if data[col1].value_counts().min() < 5 or data[col2].value_counts().min() < 5:
        raise ValueError("Some categories have less than 5 observations.")
    
    contingency_table = pd.crosstab(data[col1], data[col2])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    return p

# Example usage:
# data = pd.DataFrame({
#     'Var1': ['A'] * 40 + ['B'] * 60,
#     'Var2': ['X'] * 25 + ['Y'] * 25 + ['X'] * 25 + ['Y'] * 25
# })
# print(task_func(data, 'Var1', 'Var2'))

# np.random.seed(42)
# data = pd.DataFrame({
#     'a': np.random.choice(['A', 'B'], size=100),
#     'b': np.random.choice(['X', 'Y'], size=100)
# })
# print(task_func(data, 'a', 'b'))