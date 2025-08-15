import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler

def task_func(data, column, outlier_z_score):
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)
    column_data = data_standardized[:, column]
    z_scores = np.abs(stats.zscore(column_data))
    outlier_indices = np.where(z_scores > outlier_z_score)[0]
    data_without_outliers = np.delete(data_standardized, outlier_indices, axis=0)
    data_without_outliers = scaler.inverse_transform(data_without_outliers)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(data[:, 0], data[:, 1], color='blue', label='Data with Outliers')
    plt.scatter(data[outlier_indices, 0], data[outlier_indices, 1], color='red', label='Outliers')
    plt.title('Data with Outliers')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(data_without_outliers[:, 0], data_without_outliers[:, 1], color='green', label='Data without Outliers')
    plt.title('Data without Outliers')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    
    plt.show()
    
    return data, data_without_outliers, tuple(outlier_indices)