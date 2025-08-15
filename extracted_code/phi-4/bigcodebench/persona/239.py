import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def task_func(original):
    # Extract numeric values from the list of tuples
    numeric_values = np.array([value for _, value in original])
    
    # Compute basic statistics
    mean_val = np.mean(numeric_values)
    std_val = np.std(numeric_values)
    min_val = np.min(numeric_values)
    max_val = np.max(numeric_values)
    
    statistics = {
       'mean': mean_val,
       'std': std_val,
       'min': min_val,
       'max': max_val
    }
    
    # Generate histogram and overlay PDF
    fig, ax = plt.subplots()
    ax.hist(numeric_values, density=True, alpha=0.6, bins='auto', color='blue', edgecolor='black')
    
    # Fit a normal distribution and plot the PDF
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mean_val, std_val)
    ax.plot(x, p, 'k', linewidth=2)
    
    ax.set_title('Histogram with Normal PDF Overlay')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    
    return numeric_values, statistics, ax