import pandas as pd
from datetime import datetime
import random

def task_func(epoch_milliseconds, random_seed=0, products=["Product1", "Product2", "Product3", "Product4", "Product5"]):
    MIN_SALES = 10
    MAX_SALES = 50

    if not isinstance(epoch_milliseconds, int) or epoch_milliseconds < 0:
        raise ValueError("epoch_milliseconds must be a non-negative integer")
    
    if len(products)!= 5 or len(set(products))!= 5:
        raise ValueError("products must contain exactly 5 unique strings")
    
    epoch_datetime = datetime.fromtimestamp(epoch_milliseconds / 1000.0)
    current_datetime = datetime.now()
    
    if epoch_datetime > current_datetime:
        raise ValueError("epoch_milliseconds must be in the past")
    
    random_gen = random.Random(random_seed)
    
    date_range = pd.date_range(start=epoch_datetime.date(), end=current_datetime.date(), freq='D')
    
    sales_data = [
        {"Product": product, "Date": date, "Sales": random_gen.randint(MIN_SALES, MAX_SALES)}
        for date in date_range
        for product in products
    ]
    
    return pd.DataFrame(sales_data)