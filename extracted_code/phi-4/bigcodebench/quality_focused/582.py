import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def task_func(size=1000):
    data = np.random.normal(loc=0, scale=1, size=size)
    fig, ax = plt.subplots()

    # Plot histogram
    counts, bins, patches = ax.hist(data, bins=30, density=True, alpha=0.6, color='g')

    # Plot PDF
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, 0, 1)
    ax.plot(x, p, 'k', linewidth=2)

    return fig