import matplotlib.pyplot as plt

import numpy as np

import pandas as pd


def create_random_dataframe(columns: list, rows: int, title: str) -> tuple:

    data = np.random.randint(0, 10, size=(rows, len(columns)))

    df = pd.DataFrame(data, columns=columns)

    non_zero_counts = df.notnull().sum()

    fig, ax = plt.subplots()

    ax.bar(non_zero_counts.index, non_zero_counts.values)

    ax.set_title(title)

    plt.show()

    return df, ax


# Example usage:

columns = ['Column1', 'Column2', 'Column3', 'Column4', 'Column5']

rows = 10

title = 'Non-Zero Value Counts'

df, ax = create_random_dataframe(columns, rows, title)