import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def task_func(P, T, n_clusters=3, random_state=0, n_init=10):
    result = np.tensordot(P, T, axes=([1], [0]))
    flattened_result = result.flatten()
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    cluster_result = kmeans.fit_predict(flattened_result.reshape(-1, 1))
    
    fig, ax = plt.subplots()
    ax.scatter(flattened_result, np.zeros_like(flattened_result), c=cluster_result, cmap='viridis')
    ax.set_title('KMeans Clustering Visualization')
    
    return cluster_result, ax

# Example usage:
P = np.array([[6, 2, 7], [1, 1, 8], [8, 7, 1], [9, 6, 4], [2, 1, 1]])
T = np.random.rand(3, 3, 3)
cluster_result, ax = task_func(P, T, n_clusters=3, random_state=0, n_init=10)