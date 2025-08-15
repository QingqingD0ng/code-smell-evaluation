import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
import pandas as pd

def task_func(df, z_threshold=2):
    # Calculate Z-Scores for the 'closing_price' column
    df['z_score'] = zscore(df['closing_price'])
    
    # Identify outliers where the absolute Z-Score is greater than the threshold
    outliers = df[np.abs(df['z_score']) > z_threshold]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['closing_price'], label='Closing Price', color='blue', marker='o')
    plt.scatter(outliers.index, outliers['closing_price'], color='red', label='Outliers', zorder=5)
    
    # Setting labels and title
    plt.xlabel('Index')
    plt.ylabel('Closing Price')
    plt.title('Outliers in Closing Prices')
    plt.legend()
    
    # Return the outliers DataFrame and the plot object
    return outliers, plt.gca()