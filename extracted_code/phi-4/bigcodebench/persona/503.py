import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def task_func(days_in_past=7, stock_names=["AAPL", "GOOGL", "MSFT", "AMZN", "FB"], random_seed=0):
    np.random.seed(random_seed)
    
    # Ensure days_in_past is positive
    if days_in_past <= 0:
        raise ValueError("days_in_past must be positive")
    
    # Ensure stock_names is not empty
    if not stock_names:
        raise ValueError("stock_names must not be empty")
    
    # Generate dates
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=days_in_past - 1)
    dates = pd.date_range(start=start_date, end=end_date)
    
    # Generate random stock prices
    random_prices = np.random.random((len(dates), len(stock_names)))
    
    # Create DataFrame
    df = pd.DataFrame(random_prices, index=dates, columns=stock_names)
    
    return df