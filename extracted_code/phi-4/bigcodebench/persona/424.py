import cv2
import numpy as np
import os
from sklearn.cluster import KMeans

def task_func(image_path='image.jpg', n_clusters=3, random_seed=42):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at path: {image_path}")
    if n_clusters < 1:
        raise ValueError("n_clusters must be a positive integer.")

    # Read the image in BGR format and convert to RGB
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Reshape the image to a 2D array of pixels
    pixels = image_rgb.reshape(-1, 3)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed)
    kmeans.fit(pixels)
    clustered_pixels = kmeans.cluster_centers_[kmeans.labels_].reshape(image_rgb.shape).astype('uint8')
    
    # Save each cluster as a separate image
    for i in range(n_clusters):
        mask = (kmeans.labels_ == i).reshape(image_rgb.shape[:2])
        cluster_image = np.zeros_like(image_rgb)
        cluster_image[mask] = kmeans.cluster_centers_[i]
        cv2.imwrite(f'cluster_{i}.jpg', cv2.cvtColor(cluster_image, cv2.COLOR_RGB2BGR))
    
    return image_rgb, clustered_pixels