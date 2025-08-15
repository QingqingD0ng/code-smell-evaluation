import numpy as np
from collections import Counter
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd

def task_func(df, column_name='value', bins=4, color='green', alpha=0.6, linewidth=2):
    if df.empty or column_name not in df.columns:
        counter = Counter()
        fig, ax = plt.subplots()
        ax.set_title('Distribution')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        return counter, ax

    values = df[column_name]
    counter = Counter(value for value, count in values.value_counts().items() if count > 1)

    fig, ax = plt.subplots()
    ax.hist(values, bins=bins, color=color, alpha=alpha, density=True)

    mu, std = norm.fit(values)
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=linewidth)

    ax.set_title('Distribution')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')

    return counter, ax