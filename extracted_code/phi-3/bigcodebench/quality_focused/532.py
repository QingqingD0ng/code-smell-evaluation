import numpy as np
from collections import Counter
from scipy.stats import norm
import matplotlib.pyplot as plt

def task_func(df, bins=4):
    if df.empty:
        return Counter(), plt.figure()

    counter = Counter(df['value'])
    duplicates = {k: v for k, v in counter.items() if v > 1}

    values = df['value'].drop_duplicates()
    mean, std = np.mean(values), np.std(values)
    x = np.linspace(min(values), max(values), 100)
    y = norm.pdf(x, mean, std)

    plt.hist(df['value'], bins=bins, alpha=0.6, label='Value', color='green')
    plt.plot(x, y, 'k-', linewidth=2, label='Normal distribution')
    plt.title('Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    return Counter(duplicates), plt.gcf()