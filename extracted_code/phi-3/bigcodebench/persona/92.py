import numpy as np


def task_func(data, n_clusters=3):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame")
    if not isinstance(n_clusters, int) or n_clusters <= 1:
        raise ValueError("n_clusters must be an integer greater than 1")
    
    X = data.values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    plt.figure(figsize=(8, 6))
    collection = PathCollection(X, cmap='viridis', alpha=0.5, label='Samples')
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X')
    plt.title('K-Means Clustering')
    plt.legend(handles=collection.legend_elements()[0], title="Clusters")
    plt.xlabel('Feature1')
    plt.ylabel('Feature2')
    
    return labels, plt.gca()