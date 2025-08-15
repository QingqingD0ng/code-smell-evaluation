import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Constants
FEATURE_1 = 0
FEATURE_2 = 1

class OutlierDetector:
    def __init__(self, column, z_threshold):
        self.column = column
        self.z_threshold = z_threshold

    def standardize(self, data):
        scaler = StandardScaler()
        return scaler.fit_transform(data)

    def detect(self, data):
        if not isinstance(data, np.ndarray):
            raise ValueError('Data must be a numpy array.')
        if self.column < 0 or self.column >= data.shape[1]:
            raise ValueError('Column index out of bounds.')

        standardized_data = self.standardize(data)
        z_scores = np.abs(stats.zscore(standardized_data[:, self.column]))
        non_outliers_indices = np.where(z_scores < self.z_threshold)[0]
        outliers_indices = np.where(z_scores >= self.z_threshold)[0]
        return non_outliers_indices, outliers_indices

    def visualize(self, data, non_outliers_indices, outliers_indices=None):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(data[:, FEATURE_1], data[:, FEATURE_2])
        plt.title('Data with Outliers')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')

        plt.subplot(1, 2, 2)
        plt.scatter(data[non_outliers_indices, FEATURE_1], data[non_outliers_indices, FEATURE_2])
        if outliers_indices is not None:
            plt.scatter(data[outliers_indices