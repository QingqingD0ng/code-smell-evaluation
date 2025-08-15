import csv

from collections import Counter

import pandas as pd

import matplotlib.pyplot as plt


def task_func(file_path):

    try:

        df = pd.read_csv(file_path)

        duplicates = df.duplicated(keep=False)

        duplicates_df = df[duplicates]

        duplicates_count = duplicates_df.apply(tuple, axis=1).value_counts()

        ax = duplicates_count.plot(kind='bar')

        plt.show()

        return duplicates_count.to_dict(), ax

    except ValueError as e:

        raise ValueError("Please provide a.csv file.") from e


# Example usage:

# duplicates, ax = task_func("sample_data.csv")