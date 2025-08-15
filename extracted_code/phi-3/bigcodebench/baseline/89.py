import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler

def task_func(data, column, outlier_z_score):
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)
    z_scores = np.abs(stats.zscore(data_standardized[:, column]))
    non_outliers = data_standardized[z_scores < outlier_z_score]
    outliers = data_standardized[z_scores >= outlier_z_score]
    non_outliers_indices = np.where(z_scores < outlier_z_score)[0]
    outliers_indices = np.where(z_scores >= outlier_z_score)[0]
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(non_outliers[:, 0], non_outliers[:, 1])
    plt.title('Data with Outliers')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.subplot(1, 2, 2)
    plt.scatter(non_outliers[:, 0], non_outliers[:, 1])
    plt.scatter(outliers[:, 0], outliers[:, 1], color='r')
    plt.title('Data without Outliers')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()
    
    return data, non_outliers, (non_outliers_indices, outliers_indices)