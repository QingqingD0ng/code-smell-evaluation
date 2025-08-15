import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def task_func(original):
    values = np.array([item[1] for item in original])
    
    mean = np.mean(values)
    std_dev = np.std(values)
    minimum = np.min(values)
    maximum = np.max(values)
    statistics = {
      'mean': mean,
      'std': std_dev,
      'min': minimum,
      'max': maximum
    }
    
    fig, ax = plt.subplots()
    ax.hist(values, density=True, alpha=0.6, bins='auto', label='Histogram')
    
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mean, std_dev)
    
    ax.plot(x, p, 'k', linewidth=2, label='PDF')
    ax.legend()
    
    return values, statistics, ax