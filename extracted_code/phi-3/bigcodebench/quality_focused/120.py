import pandas as pd
from datetime import datetime, timedelta
from random import seed as random_seed

def task_func(start_date=datetime(2020, 1, 1), end_date=datetime(2020, 12, 31), seed=42):
    # Validate input types
    if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
        raise ValueError("start_date and end_date must be datetime instances")
    if start_date > end_date:
        raise ValueError("start_date must be before or equal to end_date")
    
    # Set seed for reproducibility
    random_seed(seed)
    
    # Calculate the number of days between start_date and end_date
    delta = (end_date - start_date).days + 1
    
    # Generate a series of random dates
    dates = [start_date + timedelta(days=randint(0, delta)) for _ in range(delta)]
    
    # Convert list of dates to pandas Series
    return pd.Series(dates)