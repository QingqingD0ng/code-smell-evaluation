import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler

def task_func(data, column, outlier_z_score):
    scaler = StandardScaler()
    column_data = data[:, column].reshape(-1, 1)
    standardized_data = scaler.fit_transform(column_data).flatten()
    
    z_scores = stats.zscore(standardized_data)
    outliers = np.abs(z_scores) > outlier_z_score
    non_outliers = ~outliers
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(data[:, 0], data[:, column], color='blue', label='Data with Outliers')
    plt.scatter(data[outliers, 0], data[outliers, column], color='red', label='Outliers')
    plt.title('Data with Outliers')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(data[non_outliers, 0], data[non_outliers, column], color='green', label='Data without Outliers')
    plt.title('Data without Outliers')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return data, data[non_outliers], np.where(outliers)[0]