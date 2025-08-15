import cv2
import numpy as np
import os
from sklearn.cluster import KMeans

def task_func(image_path='image.jpg', n_clusters=3, random_seed=42):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at path: {image_path}")
    
    if n_clusters < 1:
        raise ValueError("'n_clusters' must be a positive integer.")

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    original_img_array = image.copy()
    h, w, _ = image.shape
    image_data = image.reshape((-1, 3))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed)
    kmeans.fit(image_data)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    
    segmented_image_data = cluster_centers[labels].reshape(h, w, 3).astype(np.uint8)
    segmented_img_array = segmented_image_data
    
    for i in range(n_clusters):
        cluster_mask = (labels == i).reshape(h, w)
        cluster_image = np.zeros_like(image)
        cluster_image[cluster_mask] = cluster_centers[i]
        cluster_image = cv2.cvtColor(cluster_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'cluster_{i}.jpg', cluster_image)
    
    return original_img_array, segmented_img_array