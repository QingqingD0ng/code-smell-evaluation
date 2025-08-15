import pandas as pd
from datetime import datetime, timedelta
import random

def validate_products(products):
    return len(products) == 5 and len(set(products)) == len(products)

def generate_sales_data(
    epoch_milliseconds,
    random_seed=0,
    products=["Product1", "Product2", "Product3", "Product4", "Product5"],
    sales_range=(10, 50)
):
    if epoch_milliseconds >= int(datetime.now().timestamp() * 1000):
        raise ValueError("epoch_milliseconds must be before current system time")
    if not validate_products(products):
        raise ValueError("products must contain exactly 5 unique strings")

    random.seed(random_seed)

    start_date = datetime.fromtimestamp(epoch_milliseconds / 1000)
    end_date = datetime.now()
    date_range = pd.date_range(start_date, end_date)

    sales_data = []
    for date in date_range:
        for product in products:
            sales = random.randint(*sales_range)
            sales_data.append([product, date, sales])

    sales_df = pd.DataFrame(sales_data, columns=['Product', 'Date', 'Sales'])
    sales_df.sort_values(by=['Product', 'Date'], inplace=True)
    sales_df.reset_index(drop=True, inplace=True)

    return sales_df