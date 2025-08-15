import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def task_func(size=1000):
    # Generate normally distributed random numbers
    data = np.random.normal(loc=0, scale=1, size=size)
    
    # Calculate histogram
    counts, bin_edges, _ = plt.hist(data, bins=30, density=True, alpha=0.6, color='g')
    
    # Calculate PDF
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    pdf = stats.norm.pdf(bin_centers, 0, 1)
    
    # Plot PDF
    plt.plot(bin_centers, pdf, 'r-', lw=2)
    
    # Set plot labels and title
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Histogram and PDF of Normally Distributed Random Numbers')
    
    # Return the figure object
    return plt.gcf()

# Example usage:
# fig = task_func(size=500)
# isinstance(fig, matplotlib.figure.Figure)  # Check if the output is a matplotlib figure object
# len(fig.axes[0].lines) == 1  # Ensure there is one line plot on the axes for the PDF
# len(fig.axes[0].patches) > 10  # Check if there are histogram bars (patches) present