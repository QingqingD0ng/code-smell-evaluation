import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler

def task_func(data, column, outlier_z_score):
    # Standardize the specified column
    data_standardized = StandardScaler().fit_transform(data[:, column].reshape(-1, 1)).flatten()
    
    # Calculate Z-scores
    z_scores = np.abs(stats.zscore(data_standardized))
    
    # Identify outliers
    outliers = np.where(z_scores > outlier_z_score)
    outlier_indices = outliers[0]
    
    # Remove outliers from data
    data_no_outliers = np.delete(data, outlier_indices, axis=0)
    
    # Plot data with outliers
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(data[:, 0], data[:, 1], label='Data with Outliers')
    plt.scatter(data[outlier_indices, 0], data[outlier_indices, 1], color='r', label='Outliers')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    
    # Plot data without outliers
    plt.subplot(1, 2, 2)
    plt.scatter(data_no_outliers[:, 0], data_no_outliers[:, 1], label='Data without Outliers')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    
    plt.show()
    
    return data, data_no_outliers, tuple(outlier_indices)