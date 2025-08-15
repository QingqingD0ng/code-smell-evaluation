import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def task_func(data):
    values = list(map(int, data.split('-')))
    data_series = pd.Series(values)
    bins = np.arange(data_series.min(), data_series.max() + 2) - 0.5
    ax = data_series.plot(kind='hist', bins=bins, edgecolor='black')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Values')
    plt.show()
    return ax