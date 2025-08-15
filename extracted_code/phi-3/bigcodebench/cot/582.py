import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def task_func(size=1000):
    data = np.random.randn(size)
    fig, ax = plt.subplots()
    ax.hist(data, bins=30, density=True, alpha=0.5, label='Histogram')
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, np.mean(data), np.std(data))
    ax.plot(x, p, 'r', label='PDF')
    ax.legend()
    return fig