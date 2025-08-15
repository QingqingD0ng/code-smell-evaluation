import pandas as pd

import numpy as np

import matplotlib.pyplot as plt


def task_func(data):
    values = pd.Series(data.split('-')).astype(int)
    bins = np.arange(values.min(), values.max() + 2) - 0.5
    values.hist(bins=bins, ax=plt.gca())
    plt.title('Histogram of Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    return plt.gca()


# Example usage:

data = '1-2-3-4-5-6-7-8-9-10'
ax = task_func(data)