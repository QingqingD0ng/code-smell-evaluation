import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def task_func(data, n_clusters=3):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    if not isinstance(n_clusters, int) or n_clusters <= 1:
        raise ValueError("Number of clusters must be an integer greater than 1.")
    
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(data)
    
    plt.figure(figsize=(8, 6))
    centroids = kmeans.cluster_centers_
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, s=50, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75)
    plt.title('K-Means Clustering')
    plt.xlabel('Feature1')
    plt.ylabel('Feature2')
    
    return labels, plt.gca()