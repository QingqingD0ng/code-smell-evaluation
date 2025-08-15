import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler

def task_func(data, column, outlier_z_score):
    # Standardize the specified column
    scaler = StandardScaler()
    data_column = data[:, column].reshape(-1, 1)
    standardized_column = scaler.fit_transform(data_column).flatten()
    
    # Calculate Z-scores
    z_scores = np.abs(stats.zscore(standardized_column))
    
    # Identify outliers
    outlier_indices = np.where(z_scores > outlier_z_score)[0]
    
    # Remove outliers
    data_without_outliers = np.delete(data, outlier_indices, axis=0)
    
    # Plotting
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(data[:, 0], data[:, 1], color='blue', label='Data with Outliers')
    plt.scatter(data[outlier_indices, 0], data[outlier_indices, 1], color='red', label='Outliers')
    plt.title('Data with Outliers')
    plt.xlabel('Column 0')
    plt.ylabel('Column 1')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(data_without_outliers[:, 0], data_without_outliers[:, 1], color='green', label='Data without Outliers')
    plt.title('Data without Outliers')
    plt.xlabel('Column 0')
    plt.ylabel('Column 1')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return data, data_without_outliers, tuple(outlier_indices)

# Example usage
data = np.array([[14, 25], [1, 22], [7, 8], [100, 200]])
column = 1
task_func(data, column, 3.0)