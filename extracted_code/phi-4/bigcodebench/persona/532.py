import numpy as np
from collections import Counter
from scipy.stats import norm
import matplotlib.pyplot as plt

def task_func(df, bins=4):
    if df.empty or 'value' not in df.columns:
        return Counter(), plt.gca()

    values = df['value']
    value_counts = Counter(values)
    
    # Filter to get only duplicate values
    duplicates = {k: v for k, v in value_counts.items() if v > 1}
    
    # Plotting
    plt.figure()
    ax = plt.gca()
    
    # Histogram
    n, bins, patches = ax.hist(values, bins=bins, color='green', alpha=0.6, edgecolor='black')
    
    # Normal distribution overlay
    mu, std = norm.fit(values)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p * max(n) / max(p), 'k', linewidth=2)
    
    # Titles and labels
    ax.set_title('Distribution')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    
    return duplicates, ax