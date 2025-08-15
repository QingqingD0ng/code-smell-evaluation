import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.collections import PathCollection
import numpy as np

def task_func(data, n_clusters=3):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("'data' must be a pd.DataFrame.")
    if not isinstance(n_clusters, int) or n_clusters <= 1:
        raise ValueError("'n_clusters' must be an integer greater than 1.")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    
    fig, ax = plt.subplots()
    scatter = ax.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, cmap='viridis', marker='o')
    centroids = ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                           c='red', s=200, alpha=0.75, marker='x')
    
    ax.set_title('K-Means Clustering')
    handles, _ = ax.get_legend_handles_labels()
    legend1 = ax.legend(handles[0::len(labels.unique())], labels.unique(), title="Clusters")
    ax.add_artist(legend1)
    ax.legend([centroids], ['Centroids'], loc='upper right')
    
    return labels, ax