import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

def task_func(df, z_threshold=2):
    # Calculate Z-Scores
    z_scores = zscore(df['closing_price'])
    
    # Identify outliers
    outliers = df[np.abs(z_scores) > z_threshold]
    
    # Plot outliers
    plt.figure(figsize=(10, 6))
    plt.plot(df['closing_price'], label='Closing Price')
    plt.scatter(outliers.index, outliers['closing_price'], color='red', label='Outliers')
    plt.xlabel('Index')
    plt.ylabel('Closing Price')
    plt.title('Outliers in Closing Prices')
    plt.legend()
    plt.show()
    
    return outliers, plt