import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def task_func(
    P: np.ndarray,
    T: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 0,
    n_init: int = 10,
) -> (np.ndarray, plt.Axes):
    # Calculate the product of matrix 'P' and tensor 'T'
    result = np.tensordot(P, T, axes=([1],[0]))
    
    # Flatten the result
    flattened = result.flatten()
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    cluster_labels = kmeans.fit_predict(flattened.reshape(-1, 1))
    cluster_result = kmeans.cluster_centers_
    
    # Visualization
    plt.scatter(flattened, np.zeros_like(flattened), c=cluster_labels, cmap='viridis', marker='o')
    plt.scatter(cluster_result.flatten(), np.zeros_like(cluster_result), c='red', marker='x')
    plt.title('KMeans Clustering Visualization')
    ax = plt.gca()
    return cluster_result, ax