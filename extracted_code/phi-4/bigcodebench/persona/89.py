import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler

def task_func(data, column, outlier_z_score):
    original_data = data.copy()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[:, column].reshape(-1, 1)).flatten()
    z_scores = np.abs(stats.zscore(scaled_data))
    outliers = np.where(z_scores > outlier_z_score)[0]
    cleaned_data = np.delete(data, outliers, axis=0)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(data[:, 0], data[:, column], color='blue', label='Data with Outliers')
    plt.scatter(data[outliers, 0], data[outliers, column], color='red', label='Outliers')
    plt.title('Data with Outliers')
    plt.xlabel('Column 0')
    plt.ylabel(f'Column {column}')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(cleaned_data[:, 0], cleaned_data[:, column], color='green', label='Data without Outliers')
    plt.title('Data without Outliers')
    plt.xlabel('Column 0')
    plt.ylabel(f'Column {column}')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return original_data, cleaned_data, tuple(outliers)