import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_stock_prices(days_in_past=7, stock_names=["AAPL", "GOOGL", "MSFT", "AMZN", "FB"], random_seed=0):
    if not isinstance(days_in_past, int) or days_in_past <= 0:
        raise ValueError("days_in_past must be a positive integer")
    if not isinstance(stock_names, list) or not stock_names:
        raise ValueError("stock_names must be a non-empty list")
    
    np.random.seed(random_seed)
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=days_in_past - 1)
    dates = pd.date_range(start_date, end_date, freq='B')  # 'B' for business days
    
    if len(dates) < days_in_past:
        raise ValueError("The number of business days in the range is less than days_in_past")
    
    data = np.random.rand(len(dates), len(stock_names))
    df = pd.DataFrame(data, index=dates, columns=stock_names)
    return df