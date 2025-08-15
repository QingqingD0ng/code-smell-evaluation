import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap

def task_func(data, n_clusters=3):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("'data' must be a pandas DataFrame.")
    if not isinstance(n_clusters, int) or n_clusters <= 1:
        raise ValueError("'n_clusters' must be an integer greater than 1.")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data_values = data.values
    cluster_labels = kmeans.fit_predict(data_values)
    
    centroids = kmeans.cluster_centers_
    cmap = ListedColormap(plt.cm.tab10.colors[:n_clusters])
    fig, ax = plt.subplots()
    ax.scatter(data_values[:, 0], data_values[:, 1], c=cluster_labels, cmap=cmap, edgecolor='k')
    ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', label='Centroids')
    ax.legend()
    ax.set_title('K-Means Clustering')
    ax.set_xlabel(data.columns[0])
    ax.set_ylabel(data.columns[1])
    plt.show()
    
    return cluster_labels, fig