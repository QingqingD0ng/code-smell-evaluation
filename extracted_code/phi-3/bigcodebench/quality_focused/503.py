import numpy as np

import pandas as pd

from datetime import datetime, timedelta


def task_func(days_in_past=7, stock_names=["AAPL", "GOOGL", "MSFT", "AMZN", "FB"], random_seed=0):

    np.random.seed(random_seed)

    end_date = datetime.now()

    start_date = end_date - timedelta(days=days_in_past)


    date_range = pd.date_range(start=start_date, end=end_date)

    data = {stock: np.random.rand(len(date_range)) for stock in stock_names}

    df = pd.DataFrame(data, index=date_range)

    df.columns.name = 'Stock'

    df.index.name = 'Date'

    return df