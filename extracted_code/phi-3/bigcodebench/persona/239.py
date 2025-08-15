import numpy as np

import matplotlib.pyplot as plt

from scipy import stats


def task_func(original):

    arr = np.array([t[1] for t in original])

    mean = np.mean(arr)

    std = np.std(arr)

    min_val = np.min(arr)

    max_val = np.max(arr)


    stats_dict = {

       'mean': mean,

       'std': std,

       'min': min_val,

       'max': max_val

    }


    norm_dist = stats.norm(mean, std)

    x = np.linspace(min_val, max_val, 100)

    y = norm_dist.pdf(x)


    fig, ax = plt.subplots()

    ax.hist(arr, bins='auto', density=True, alpha=0.6, color='g', edgecolor='black')

    ax.plot(x, y, 'r', linewidth=2)


    return arr, stats_dict, ax