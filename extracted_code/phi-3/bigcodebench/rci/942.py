import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

def generate_date_range(start_date: str, periods: int, freq: str) -> pd.DatetimeIndex:
    """
    Generate a date range given the start date, periods, and frequency.

    :param start_date: Start date in 'YYYY-MM-DD' format.
    :param periods: Number of periods.
    :param freq: Date frequency.
    :return: Date range as a pandas DatetimeIndex object.
    """
    return pd.date_range(start=start_date, periods=periods, freq=freq)

def generate_sales_data(categories: List[str], date_range: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Generate random sales data for each category over a given date range.

    :param categories: List of category names.
    :param date_range: Date range as a pandas DatetimeIndex object.
    :return: DataFrame with sales data.
    """
    np.random.seed(0)
    sales_data = {category: np.random.randint(low=50, high=500, size=len(date_range)) for category in categories}
    sales_df = pd.DataFrame(sales_data, index=date_range)
    sales_df.reset_index(inplace=True)
    sales_df.rename(columns={'index': 'Date'}, inplace=True)
    sales_df['Sales'] = sales_df.drop('Category', axis=1).sum(axis=1)
    return sales_df

def task_func(start_date: str = '2016-01-01', periods: int = 13, freq: str = 'WOM-2FRI', categories: List[str] = ['Electronics', 'Fashion', 'Home & Kitchen', 'Automotive', 'Sports']) -> Tuple[pd.DataFrame, plt.Axes]:
    """
    Create and visualize a sales report for different categories over a period of time