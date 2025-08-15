import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import zscore


def calculate_z_scores(df):
    """Calculate and add z-scores to the DataFrame."""
    df['z_score'] = zscore(df['closing_price'])
    return df


def identify_outliers(df, threshold):
    """Identify and return the DataFrame of outliers."""
    return df[np.abs(df['z_score']) > threshold]


def plot_outliers(df, outliers, title):
    """Plot the closing prices with outliers highlighted."""
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['closing_price'], label='Closing Price')
    plt.scatter(outliers.index, outliers['closing_price'], color='red', label='Outliers')
    plt.xlabel('Index')
    plt.ylabel('Closing Price')
    plt.title(title)
    plt.legend()


def task_func(df, z_threshold=2):
    df = calculate_z_scores(df)
    outliers = identify_outliers(df, z_threshold)
    plot_outliers(df, outliers, 'Outliers in Closing Prices')
    return outliers, plt