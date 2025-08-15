import pandas as pd

from datetime import datetime, timedelta

import random


def task_func(
    epoch_milliseconds,
    random_seed=0,
    products=["Product1", "Product2", "Product3", "Product4", "Product5"],
):

    random.seed(random_seed)

    start_date = datetime.utcfromtimestamp(epoch_milliseconds / 1000)

    end_date = datetime.utcnow()


    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    data = {

        'Product': [],

        'Date': [],

        'Sales': []

    }


    for date in dates:

        for product in products:

            sales = random.randint(10, 50)

            data['Product'].append(product)

            data['Date'].append(date)

            data['Sales'].append(sales)


    sales_data = pd.DataFrame(data)

    return sales_data