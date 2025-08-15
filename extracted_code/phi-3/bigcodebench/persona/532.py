import numpy as np
from collections import Counter
from scipy.stats import norm
import matplotlib.pyplot as plt

def task_func(df, bins=4):
    if df.empty:
        return Counter(), plt.gca()

    value_counts = Counter(df['value'])
    duplicates = {k: v for k, v in value_counts.items() if v > 1}
    
    value_list = df['value'].drop_duplicates().sort_values().tolist()
    plt.hist(value_list, bins=bins, color='green', alpha=0.6, label='Data')
    
    mu, std = norm.fit(value_list)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p * (xmax - xmin) / len(value_list) + xmin, 'k', linewidth=2, label='Normal fit')

    plt.title('Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()

    return Counter(duplicates), plt.gca()