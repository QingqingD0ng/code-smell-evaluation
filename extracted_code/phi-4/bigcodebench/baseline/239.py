import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def task_func(original):
    # Extract numeric values from the tuples
    numeric_values = np.array([item[1] for item in original])
    
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
    
    # Generate histogram with overlaid PDF
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    counts, bins, patches = ax.hist(numeric_values, density=True, alpha=0.6, bins='auto')
    
    # Fit a normal distribution to the data
    mu, sigma = stats.norm.fit(numeric_values)
    pdf_x = np.linspace(min_val, max_val, 100)
    pdf_y = stats.norm.pdf(pdf_x, mu, sigma)
    
    # Plot the PDF
    ax.plot(pdf_x, pdf_y, 'r-', lw=2)
    
    return numeric_values, statistics, ax