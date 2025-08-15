import pandas as pd

import numpy as np

import matplotlib.pyplot as plt


def task_func(data):
    values = pd.Series(data.split('-')).astype(float)
    bins = np.arange(values.min(), values.max() + 2) - 0.5
    ax = values.hist(bins=bins)
    ax.set_title('Histogram of Values')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    return ax