import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def task_func(days_in_past=7, stock_names=["AAPL", "GOOGL", "MSFT", "AMZN", "FB"], random_seed=0):
    np.random.seed(random_seed)
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=days_in_past - 1)
    dates = pd.date_range(start_date, end_date)
    data = np.random.rand(days_in_past, len(stock_names))
    df = pd.DataFrame(data, index=dates, columns=stock_names)
    return df