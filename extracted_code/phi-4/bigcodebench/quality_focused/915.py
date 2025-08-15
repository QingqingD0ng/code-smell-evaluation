import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore

def task_func(df, z_threshold=2):
    z_scores = zscore(df['closing_price'])
    outliers = df[np.abs(z_scores) > z_threshold]
    
    plt.figure()
    plt.plot(df.index, df['closing_price'], label='Closing Price')
    plt.scatter(outliers.index, outliers['closing_price'], color='red', label='Outliers')
    plt.xlabel('Index')
    plt.ylabel('Closing Price')
    plt.title('Outliers in Closing Prices')
    plt.legend()
    
    return outliers, plt.gca()