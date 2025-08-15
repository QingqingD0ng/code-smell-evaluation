import numpy as np

from scipy import stats

import matplotlib.pyplot as plt


def task_func(size=1000):
    data = np.random.randn(size)
    fig, ax = plt.subplots()

    # Plot histogram
    ax.hist(data, bins=30, density=True, alpha=0.7, color='gray')

    # Calculate and plot PDF
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, 0, 1)
    ax.plot(x, p, 'k', linewidth=2)

    return fig


# Example usage

fig = task_func(size=500)