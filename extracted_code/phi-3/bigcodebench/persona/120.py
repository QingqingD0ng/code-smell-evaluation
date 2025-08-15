import pandas as pd
from datetime import datetime, timedelta
from random import seed as random_seed

def task_func(start_date=datetime(2020, 1, 1), end_date=datetime(2020, 12, 31), seed=42):
    if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
        raise ValueError("start_date and end_date must be datetime instances")
    if start_date > end_date:
        raise ValueError("start_date must be before end_date")
    
    random_seed(seed)
    
    date_range = pd.date_range(start=start_date, end=end_date).to_series()
    random_indices = random_seed.randint(0, len(date_range) - 1, size=len(date_range))
    random_dates = date_range.iloc[random_indices].reset_index(drop=True)
    
    return random_dates