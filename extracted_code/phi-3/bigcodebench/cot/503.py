import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def task_func(days_in_past=7, stock_names=None, random_seed=0):
    if stock_names is None:
        stock_names = ["AAPL", "GOOGL", "MSFT", "AMZN", "FB"]
    if not stock_names:
        raise ValueError("stock_names must not be empty")
    if days_in_past <= 0:
        raise ValueError("days_in_past must be positive")

    np.random.seed(random_seed)
    start_date = datetime.now() - timedelta(days=days_in_past)
    dates = [start_date + timedelta(days=i) for i in range(days_in_past)]
    data = {name: np.random.rand(days_in_past).cumprod() for name in stock_names}
    df = pd.DataFrame(data, index=pd.to_datetime(dates))
    df.index.name = 'Date'
    return df

# Example usage
df = task_func(5, random_seed=42)
print(df.head(1))