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
    # Validate input shapes
    if P.ndim!= 2 or T.ndim!= 3:
        raise ValueError("P must be a 2D array and T must be a 3D array.")
    
    # Calculate the product of matrix P and tensor T
    product_result = np.tensordot(P, T, axes=([1], [0]))
    
    # Flatten the result
    flattened_data = product_result.reshape(product_result.shape[0], -1)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    cluster_result = kmeans.fit_predict(flattened_data)
    
    # Visualize the clustering
    if flattened_data.shape[1] < 2:
        raise ValueError("Flattened data must have at least two dimensions for scatter plot.")
    
    fig, ax = plt.subplots()
    ax.scatter(flattened_data[:, 0], flattened_data[:, 1], c=cluster_result, cmap='viridis')
    ax.set_title('KMeans Clustering Visualization')
    
    return cluster_result, ax