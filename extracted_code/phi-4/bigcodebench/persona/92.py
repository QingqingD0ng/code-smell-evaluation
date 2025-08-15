import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

def task_func(data, n_clusters=3):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    if not isinstance(n_clusters, int) or n_clusters <= 1:
        raise ValueError("'n_clusters' must be an integer greater than 1.")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    centroids = kmeans.cluster_centers_

    fig, ax = plt.subplots()
    scatter = ax.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, cmap='viridis', marker='o')
    centroids_scatter = ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100)

    ax.set_title('K-Means Clustering')
    ax.legend([scatter, centroids_scatter], ['Clusters', 'Centroids'])
    plt.show()

    return labels, ax