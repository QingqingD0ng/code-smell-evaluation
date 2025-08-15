import numpy as np

import matplotlib.pyplot as plt

import pandas as pd


def task_func(elements, seed=0):
    if not isinstance(elements, int) or elements <= 0:
        raise ValueError("elements must be a positive integer")

    np.random.seed(seed)
    steps = np.random.choice([-1, 1], size=elements)
    position = np.cumsum(steps)

    descriptive_stats = pd.Series(position).describe().loc[['count','mean','std','min', '25%', '50%', '75%', '95%']]
    descriptive_stats.name = 'descriptive_stats'
    descriptive_stats['max'] = position.max()

    fig, ax = plt.subplots()
    ax.plot(position)
    ax.set_title('Random Walk')
    ax.set_xlabel('Step')
    ax.set_ylabel('Position')

    return descriptive_stats, ax


# Example usage:

stats, ax = task_func(1000)
print(stats)