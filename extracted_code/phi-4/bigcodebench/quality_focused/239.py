import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def task_func(original):
    # Extract numeric values from the list of tuples
    numeric_values = np.array([x[1] for x in original])

    # Calculate basic statistics
    statistics = {
       'mean': np.mean(numeric_values),
       'std': np.std(numeric_values),
       'min': np.min(numeric_values),
       'max': np.max(numeric_values)
    }

    # Plot histogram with overlaid PDF
    fig, ax = plt.subplots()
    ax.hist(numeric_values, density=True, alpha=0.6, bins='auto')

    # Fit a normal distribution to the data
    mu, std = stats.norm.fit(numeric_values)
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2)

    return numeric_values, statistics, ax