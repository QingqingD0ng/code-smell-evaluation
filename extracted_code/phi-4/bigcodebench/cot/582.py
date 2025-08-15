import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def task_func(size=1000):
    data = np.random.normal(loc=0, scale=1, size=size)
    counts, bin_edges, patches = plt.hist(data, bins=30, density=True, alpha=0.6, color='g')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    pdf = stats.norm.pdf(x, 0, 1)
    plt.plot(x, pdf, 'k', linewidth=2)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Histogram and PDF of Normally Distributed Data')
    fig = plt.gcf()
    return fig