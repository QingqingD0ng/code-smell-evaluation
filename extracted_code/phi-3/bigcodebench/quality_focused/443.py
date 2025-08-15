import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def task_func(P: np.ndarray, T: np.ndarray, n_clusters: int = 3, random_state: int = 0, n_init: int = 10) -> (np.ndarray, plt.Axes):
    result = P.dot(T.reshape(P.shape[0], -1)).flatten()
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init).fit(result.reshape(-1, 1))
    ax = plt.scatter(result, np.zeros_like(result), c=kmeans.labels_, cmap='viridis')
    plt.title('KMeans Clustering Visualization')
    plt.xlabel('Clustered Values')
    plt.yticks([])
    return kmeans.cluster_centers_, ax