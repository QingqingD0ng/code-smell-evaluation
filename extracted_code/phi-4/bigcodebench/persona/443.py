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
    # Calculate the product of matrix P and tensor T
    result = np.tensordot(P, T, axes=([1], [0]))

    # Flatten the result
    flattened_result = result.reshape(-1, result.shape[-1])

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    cluster_result = kmeans.fit_predict(flattened_result)

    # Visualize the clustering
    fig, ax = plt.subplots()
    scatter = ax.scatter(flattened_result[:, 0], flattened_result[:, 1], c=cluster_result, cmap='viridis')
    plt.title('KMeans Clustering Visualization')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)

    return cluster_result, ax