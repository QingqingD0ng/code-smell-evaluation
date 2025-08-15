import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import zscore


def task_func(df, z_threshold=2):
    df['z_score'] = zscore(df['closing_price'])
    outliers = df[np.abs(df['z_score']) > z_threshold]
    plt.figure(figsize=(10, 6))
    plt.plot(df['closing_price'], 'o', label='Data points')
    plt.plot(outliers.index, outliers['closing_price'], 'ro', label='Outliers')
    plt.xlabel('Index')
    plt.ylabel('Closing Price')
    plt.title('Outliers in Closing Prices')
    plt.legend()
    plt.show()
    return outliers, plt