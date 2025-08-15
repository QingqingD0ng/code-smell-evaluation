import pandas as pd

from scipy.stats import chi2_contingency


def task_func(data, col1, col2):

    if data.empty:

        raise ValueError("Data is empty")

    if col1 not in data.columns or col2 not in data.columns:

        raise ValueError(f"Column names {col1} or {col2} not in data")

    if not pd.api.types.is_categorical_dtype(data[col1]) or not pd.api.types.is_categorical_dtype(data[col2]):

        raise TypeError("One or both columns are not categorical")

    contingency_table = pd.crosstab(data[col1], data[col2])

    chi2, p, dof, expected = chi2_contingency(contingency_table)

    if any(expected.sum(axis=1) < 5):

        raise ValueError("Some categories have less than 5 observations")

    return p