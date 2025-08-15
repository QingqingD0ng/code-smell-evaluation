import pandas as pd
from datetime import datetime, timedelta
import random

def task_func(epoch_milliseconds, random_seed=0, products=None):
    if products is None:
        products = ["Product1", "Product2", "Product3", "Product4", "Product5"]

    random.seed(random_seed)

    end_time = datetime.utcnow()
    start_time = datetime.utcfromtimestamp(epoch_milliseconds / 1000)

    if start_time > end_time:
        raise ValueError("epoch_milliseconds must be before current time.")

    all_data = []
    current_time = start_time
    while current_time < end_time:
        date_data = {
            "Product": random.choice(products),
            "Date": current_time,
            "Sales": random.randint(10, 50)
        }
        all_data.append(date_data)
        current_time += timedelta(days=1)

    sales_data = pd.DataFrame(all_data)
    
    return sales_data