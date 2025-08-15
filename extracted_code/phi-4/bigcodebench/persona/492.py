import pandas as pd
from datetime import datetime
import random

def task_func(epoch_milliseconds, random_seed=0, products=["Product1", "Product2", "Product3", "Product4", "Product5"]):
    # Validate inputs
    if not isinstance(epoch_milliseconds, int):
        raise ValueError("epoch_milliseconds must be an integer.")
    if not all(isinstance(product, str) for product in products):
        raise ValueError("All products must be strings.")
    if len(products)!= len(set(products)):
        raise ValueError("Products must contain 5 unique strings.")
    if len(products)!= 5:
        raise ValueError("Products list must contain exactly 5 products.")

    # Set random seed
    random.seed(random_seed)

    # Convert epoch milliseconds to datetime
    start_date = datetime.fromtimestamp(epoch_milliseconds / 1000.0)

    # Get current datetime
    end_date = datetime.now()

    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Initialize list to store sales data
    sales_data = []

    # Generate sales data for each date
    for single_date in date_range:
        for product in products:
            sales_quantity = random.randint(10, 50)
            sales_data.append([product, single_date, sales_quantity])

    # Create DataFrame
    df = pd.DataFrame(sales_data, columns=['Product', 'Date', 'Sales'])

    return df