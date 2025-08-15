import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def task_func(elements, seed=0):
    if elements <= 0 or not isinstance(elements, int):
        raise ValueError("elements must be a positive integer")

    np.random.seed(seed)
    steps = np.random.choice([-1, 1], size=elements)
    random_walk = np.cumsum(steps)

    stats = pd.Series(random_walk).describe(percentiles=[.05,.25,.5,.75,.95])
    stats_dict = {
        'count': stats['count'],
       'mean': stats['mean'],
       'std': stats['std'],
       'min': stats['min'],
        '5%': stats['5%'],
        '25%': stats['25%'],
        '50%': stats['50%'],
        '75%': stats['75%'],
        '95%': stats['95%'],
       'max': stats['max']
    }

    plt.figure()
    ax = plt.plot(random_walk)
    plt.title("Random Walk")
    plt.xlabel("Steps")
    plt.ylabel("Position")

    return stats_dict, ax