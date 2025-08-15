import matplotlib.pyplot as plt

import numpy as np

import pandas as pd


def task_func(rows):
    data = np.random.randint(0, 10, size=(rows, 5))
    df = pd.DataFrame(data, columns=COLUMNS)
    non_zero_counts = df.count()
    ax = non_zero_counts.plot(kind='bar')
    ax.set_title('Non-Zero Value Counts')
    return df, ax


# Example usage:

df, ax = task_func(10)