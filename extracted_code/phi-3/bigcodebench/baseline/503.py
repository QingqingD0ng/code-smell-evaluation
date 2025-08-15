import numpy as np

import pandas as pd

from datetime import datetime, timedelta


def task_func(
    days_in_past=7, stock_names=["AAPL", "GOOGL", "MSFT", "AMZN", "FB"], random_seed=0
):

    np.random.seed(random_seed)

    last_date = datetime.now() - timedelta(days=days_in_past)

    dates = pd.date_range(start=last_date, periods=days_in_past + 1)

    data = {stock: np.random.rand(days_in_past + 1).tolist() for stock in stock_names}

    df = pd.DataFrame(data, index=dates)

    df = df.apply(lambda x: (x - x.min()) / (x.max() - x.min())).astype(float)

    return df