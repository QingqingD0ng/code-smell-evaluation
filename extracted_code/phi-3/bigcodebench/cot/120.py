import pandas as pd
import datetime
from random import seed as random_seed

def task_func(start_date=datetime(2020, 1, 1), end_date=datetime(2020, 12, 31), seed=42):
    if not isinstance(start_date, datetime.datetime) or not isinstance(end_date, datetime.datetime):
        raise ValueError("start_date and end_date must be datetime instances")
    if start_date > end_date:
        raise ValueError("start_date must be before or equal to end_date")

    random_seed(seed)
    date_range = pd.date_range(start=start_date, end=end_date + timedelta(days=1))
    random_dates = [random_date for random_date in date_range]
    random.shuffle(random_dates)
    return pd.Series(random_dates, name='Random Dates')