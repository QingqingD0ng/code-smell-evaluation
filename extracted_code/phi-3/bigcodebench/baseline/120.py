import pandas as pd
from datetime import datetime, timedelta
import random

def task_func(start_date=datetime(2020, 1, 1), end_date=datetime(2020, 12, 31), seed=42):
    if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
        raise ValueError("start_date and end_date must be datetime instances")
    if start_date > end_date:
        raise ValueError("start_date must be before or equal to end_date")
    
    random.seed(seed)
    date_range = [start_date + timedelta(days=randint(0, (end_date - start_date).days)) for _ in range((end_date - start_date).days + 1)]
    return pd.Series(date_range)