import numpy as np
from collections import Counter
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd

def task_func(df, bins=4):
    if df.empty or 'value' not in df:
        counter = Counter()
        fig, ax = plt.subplots()
        ax.set_title('Distribution')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        return counter, ax

    values = df['value']
    counter = Counter([x for x in values if list(values).count(x) > 1])

    fig, ax = plt.subplots()
    ax.hist(values, bins=bins, color='green', alpha=0.6, density=True)

    mu, std = norm.fit(values)
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2)

    ax.set_title('Distribution')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')

    return counter, ax