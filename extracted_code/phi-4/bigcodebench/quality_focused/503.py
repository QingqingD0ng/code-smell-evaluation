import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def task_func(days_in_past=7, stock_names=["AAPL", "GOOGL", "MSFT", "AMZN", "FB"], random_seed=0):
    np.random.seed(random_seed)
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_in_past)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    random_prices = np.random.rand(len(date_range), len(stock_names))
    df = pd.DataFrame(random_prices, index=date_range, columns=stock_names)
    return df