import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def task_func(original):
    values = np.array([t[1] for t in original])
    stats_dict = {
      'mean': np.mean(values),
      'std': np.std(values, ddof=1),
      'min': np.min(values),
      'max': np.max(values)
    }
    
    kde = stats.gaussian_kde(values)
    x = np.linspace(values.min(), values.max(), 200)
    ax = plt.hist(values, bins='auto', density=True, alpha=0.6, label='Data Histogram')
    plt.plot(x, kde(x), label='KDE', color='red')
    plt.legend()
    
    return values, stats_dict, ax

# Example usage:
original = [('a', 1), ('b', 2), ('c', 3), ('d', 4)]
arr, stats, ax = task_func(original)
plt.show()