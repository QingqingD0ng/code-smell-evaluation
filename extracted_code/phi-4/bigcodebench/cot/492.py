import pandas as pd
from datetime import datetime, timedelta
import random

def task_func(epoch_milliseconds, random_seed=0, products=["Product1", "Product2", "Product3", "Product4", "Product5"]):
    if len(products)!= 5 or len(set(products))!= 5:
        raise ValueError("Products list must contain 5 unique strings.")
    
    start_date = datetime.utcfromtimestamp(epoch_milliseconds / 1000)
    current_date = datetime.utcnow()
    
    if start_date >= current_date:
        raise ValueError("Epoch time must be before the current system time.")
    
    random.seed(random_seed)
    
    all_dates = [start_date + timedelta(days=x) for x in range((current_date - start_date).days + 1)]
    sales_data = []

    for date in all_dates:
        for product in products:
            sales = random.randint(10, 50)
            sales_data.append([product, date, sales])
    
    df = pd.DataFrame(sales_data, columns=["Product", "Date", "Sales"])
    return df