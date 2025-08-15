import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler

def task_func(data, column, outlier_z_score):
    original_data = data.copy()
    column_data = data[:, column]
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(column_data.reshape(-1, 1)).flatten()
    z_scores = np.abs(stats.zscore(standardized_data))
    non_outliers = standardized_data > outlier_z_score
    clean_data = data[non_outliers]
    outliers_indices = np.where(~non_outliers)[0]
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(range(len(standardized_data)), standardized_data)
    plt.title('Data with Outliers')
    plt.xlabel('Index')
    plt.ylabel('Standardized Value')
    
    plt.subplot(1, 2, 2)
    plt.scatter(range(len(clean_data[:, column])), clean_data[:, column])
    plt.title('Data without Outliers')
    plt.xlabel('Index')
    plt.ylabel('Value')
    
    plt.tight_layout()
    plt.show()
    
    return original_data, clean_data, tuple(outliers_indices)