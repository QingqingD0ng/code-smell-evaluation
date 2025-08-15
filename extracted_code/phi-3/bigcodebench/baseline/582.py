import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def task_func(size=1000):
    data = np.random.normal(size=size)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].hist(data, bins=30, density=True)
    xmin, xmax = axs[0].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, 0, 1)
    axs[0].plot(x, p, 'k', linewidth=2)
    axs[1].hist(data, bins=30, density=True, alpha=0.5)
    axs[1].plot(x, p, 'k', linewidth=2)
    return fig

# Example usage:
# import matplotlib
# fig = task_func(size=500)
# isinstance(fig, matplotlib.figure.Figure)
# len(fig.axes[0].lines) == 1
# len(fig.axes[0].patches) > 10