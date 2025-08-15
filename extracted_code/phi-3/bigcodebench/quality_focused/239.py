import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def task_func(original):
    # Extract numeric values
    values = np.array([t[1] for t in original])
    
    # Compute basic statistics
    stats_dict = {
       'mean': np.mean(values),
       'std': np.std(values),
       'min': np.min(values),
       'max': np.max(values)
    }
    
    # Generate histogram with overlaid PDF
    plt.hist(values, bins='auto', density=True, alpha=0.6, color='g', label='Histogram')
    
    # Fit a normal distribution to the data and plot it
    params = stats.norm.fit(values)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, *params)
    
    plt.plot(x, p, 'k', linewidth=2, label='Fit PDF')
    plt.legend()
    
    ax = plt.gca()  # Get current axes
    return values, stats_dict, ax

# Example usage
original = [('a', 1), ('b', 2), ('c', 3), ('d', 4)]
arr, stats, ax = task_func(original)
plt.show()  # Display the plot

# Printing the statistics
print(arr)
print(stats)