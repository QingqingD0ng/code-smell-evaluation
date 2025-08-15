import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def run_kmeans_clustering(data: np.ndarray, num_clusters: int = 3, random_state: int = 0) -> np.ndarray:
    # Reshape the data tensor to a 2D array for dot product
    reshaped_data = data.reshape(-1, data.shape[-1])
    # Flatten the reshaped data for KMeans
    flattened_data = reshaped_data.flatten()
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(flattened_data.reshape(-1, 1))
    
    # Visualize the clustering
    plt.scatter(range(len(cluster_labels)), flattened_data, c=cluster_labels, cmap='viridis')
    plt.title('KMeans Clustering Visualization')
    plt.xlabel('Data Point Index')
    plt.ylabel('Clustered Value')
    plt.show()
    
    return cluster_labels

# Example usage with dummy data
if __name__ == "__main__":
    P = np.array([[6, 2, 7], [1, 1, 8], [8, 7, 1], [9, 6, 4], [2, 1, 1]])
    T = np.random.rand(3, 3, 3)  # Example tensor with random values
    run_kmeans_clustering(np.dot(P, T.reshape(T.shape[0], -1)), n_clusters=3, random_state=0)