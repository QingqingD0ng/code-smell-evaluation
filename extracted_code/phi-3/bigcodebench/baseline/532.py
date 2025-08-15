import numpy as np
from collections import Counter
from scipy.stats import norm
import matplotlib.pyplot as plt

def task_func(df, bins=4):
    if df.empty:
        return Counter(), plt.figure().add_subplot(111)
    value_counts = Counter(df['value'].values)
    duplicates = {val: count for val, count in value_counts.items() if count > 1}
    ax = plt.hist(df['value'], bins=bins, color='green', alpha=0.6)
    plt.title('Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    mu, std = norm.fit(df['value'])
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    plt.show()
    return duplicates, ax