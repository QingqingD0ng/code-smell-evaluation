import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def task_func(size=1000, mean=0, std_dev=1, bins=30):
    data = np.random.normal(loc=mean, scale=std_dev, size=size)
    fig, (ax_hist, ax_pdf) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Plot histogram
    ax_hist.hist(data, bins=bins, density=True, alpha=0.7, color='blue', edgecolor='black')
    xmin, xmax = ax_hist.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    ax_hist.plot(x, stats.norm.pdf(x, mean, std_dev), 'k', linewidth=2)

    # Plot PDF
    ax_pdf.hist(data, bins=bins, density=True, alpha=0.4, color='green', edgecolor='black')
    ax_pdf.plot(x, stats.norm.pdf(x, mean, std_dev), 'k', linewidth=2)

    fig.suptitle('Normal Distribution Histogram and PDF')
    return fig

# Example usage:
# import matplotlib
# fig = task_func(size=500, mean=0, std_dev=1, bins=30)
# isinstance(fig, matplotlib.figure.Figure)
# len(fig.axes[0].lines) == 1
# len(fig.axes[0].patches) > 10