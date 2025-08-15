import numpy as np

import pandas as pd

from datetime import datetime, timedelta


def generate_random_prices(num_days, stock_names):

    if not isinstance(num_days, int) or num_days <= 0:

        raise ValueError("Number of days must be a positive integer.")

    if not isinstance(stock_names, list) or not stock_names:

        raise ValueError("Stock names must be a non-empty list.")


    last_date = datetime.now() - timedelta(days=num_days)

    dates = pd.date_range(start=last_date, periods=num_days + 1)

    data = {stock: np.random.rand(num_days + 1).tolist() for stock in stock_names}

    df = pd.DataFrame(data, index=dates)

    df = (df - df.min()) / (df.max() - df.min())

    return df