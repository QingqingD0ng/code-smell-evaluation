import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler

def task_func(data, column, outlier_z_score):
    # Standardize the specified column
    scaler = StandardScaler()
    data_column = data[:, column].reshape(-1, 1)
    standardized_column = scaler.fit_transform(data_column)
    
    # Calculate Z-scores for the standardized column
    z_scores = np.abs(stats.zscore(standardized_column, axis=0))
    
    # Identify indices of outliers
    outlier_indices = np.where(z_scores > outlier_z_score)[0]
    
    # Data without outliers
    data_no_outliers = np.delete(data, outlier_indices, axis=0)
    
    # Plot data with outliers
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(data[:, 0], data[:, 1], color='blue', label='Data Points')
    plt.scatter(data[outlier_indices, 0], data[outlier_indices, 1], color='red', label='Outliers')
    plt.title('Data with Outliers')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    # Plot data without outliers
    plt.subplot(1, 2, 2)
    plt.scatter(data_no_outliers[:, 0], data_no_outliers[:, 1], color='green', label='Data Points')
    plt.title('Data without Outliers')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return data, data_no_outliers, tuple(outlier_indices)