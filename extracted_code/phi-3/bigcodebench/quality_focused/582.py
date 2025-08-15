import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def task_func(size=1000):
    data = np.random.normal(loc=0, scale=1, size=size)
    fig, ax = plt.subplots()
    count, bins, ignored = ax.hist(data, bins=30, density=True, alpha=0.5, color='g', label='Histogram')
    pdf = stats.norm.pdf(bins, np.mean(data), np.std(data))
    ax.plot(bins, pdf, linewidth=2, color='r', label='PDF')
    ax.legend()
    return fig