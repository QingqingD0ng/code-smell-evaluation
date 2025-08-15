import numpy as np

import pandas as pd

from datetime import datetime, timedelta


def task_func(days_in_past=7, stock_names=["AAPL", "GOOGL", "MSFT", "AMZN", "FB"], random_seed=0):

    np.random.seed(random_seed)

    end_date = datetime.now()

    start_date = end_date - timedelta(days=days_in_past)

    dates = pd.date_range(start_date, end_date, freq='D')

    data = {

        stock: np.random.random(size=len(dates)) for stock in stock_names

    }

    df = pd.DataFrame(data, index=dates)

    df.columns = stock_names

    df.index.name = 'Date'

    return df


# Example usage:

# df = task_func(5, random_seed=42)