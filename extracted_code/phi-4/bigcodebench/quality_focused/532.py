import numpy as np
from collections import Counter
from scipy.stats import norm
import matplotlib.pyplot as plt

def task_func(df, bins=4):
    if df.empty or 'value' not in df.columns:
        return Counter(), None

    values = df['value']
    duplicates = [item for item, count in Counter(values).items() if count > 1]
    duplicate_counts = Counter(values)
    duplicate_counts = {k: v for k, v in duplicate_counts.items() if v > 1}

    plt.figure()
    ax = plt.gca()
    n, bins, patches = ax.hist(values, bins=bins, color='green', alpha=0.6, density=True)

    mu, std = norm.fit(values)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2)

    ax.set_title('Distribution')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')

    return Counter(duplicate_counts), ax