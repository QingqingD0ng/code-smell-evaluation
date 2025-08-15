import cv2
import numpy as np
import os
from sklearn.cluster import KMeans

def task_func(image_path='image.jpg', n_clusters=3, random_seed=42):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"No file found at the specified path: {image_path}")
    if n_clusters < 1:
        raise ValueError("'n_clusters' must be a positive integer")

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image_array = image_rgb.reshape((-1, 3))

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed).fit(original_image_array)
    segmented_image_array = kmeans.cluster_centers_[kmeans.labels_].reshape(image_rgb.shape).astype(np.uint8)

    return original_image_array, segmented_image_array