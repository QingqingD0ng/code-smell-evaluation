import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def task_func(original):
    arr = np.array([t[1] for t in original])
    stats = {
       'mean': np.mean(arr),
       'std': np.std(arr),
       'min': np.min(arr),
       'max': np.max(arr)
    }
    mu, sigma = stats.norm.fit(arr)
    n, bins, patches = plt.hist(arr, density=True, alpha=0.6, bins='auto', label='Histogram')
    y = stats.norm.pdf(bins, mu, sigma)
    plt.plot(bins, y, 'r--', linewidth=2, label='PDF')
    plt.legend()
    ax = plt.gca()
    return arr, stats, ax