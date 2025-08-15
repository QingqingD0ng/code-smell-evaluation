import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def task_func(data):
    values = list(map(int, data.split('-')))
    series = pd.Series(values)
    bins = np.arange(series.min(), series.max() + 2) - 0.5
    ax = series.plot.hist(bins=bins, title='Histogram of Values', edgecolor='black')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    return ax