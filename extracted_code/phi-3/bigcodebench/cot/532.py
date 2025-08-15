import numpy as np
from collections import Counter
from scipy.stats import norm
import matplotlib.pyplot as plt

def task_func(df, bins=4):
    if df.empty:
        return Counter(), plt.figure().add_subplot(111)
    
    value_counts = df['value'].value_counts()
    duplicates = value_counts[value_counts > 1]
    counter = Counter(duplicates.index)

    fig, ax = plt.subplots()
    ax.hist(df['value'], bins=bins, alpha=0.6, color='green')
    mu, std = norm.fit(df['value'])
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2)
    ax.set_title('Distribution')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')

    return counter, ax