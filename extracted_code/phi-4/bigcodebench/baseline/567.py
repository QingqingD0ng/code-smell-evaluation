import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def task_func(data):
    values = list(map(int, data.split('-')))
    df = pd.DataFrame(values, columns=['Value'])
    bins = np.arange(df['Value'].min(), df['Value'].max() + 2) - 0.5
    ax = df['Value'].hist(bins=bins, edgecolor='black')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Values')
    return ax