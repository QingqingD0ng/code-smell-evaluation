import cv2
import numpy as np
import os
from sklearn.cluster import KMeans

def task_func(image_path='image.jpg', n_clusters=3, random_seed=42):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File {image_path} does not exist.")

    if not isinstance(n_clusters, int) or n_clusters < 1:
        raise ValueError("'n_clusters' must be a positive integer.")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at {image_path} could not be read.")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image.copy()

    h, w, d = image.shape
    image_reshaped = image.reshape(-1, d)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed)
    kmeans.fit(image_reshaped)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    segmented_image = centers[labels].reshape(h, w, d).astype(np.uint8)

    for i in range(n_clusters):
        cluster_image = np.zeros_like(image)
        cluster_image[labels == i] = centers[i]
        cv2.imwrite(f'cluster_{i}.jpg', cv2.cvtColor(cluster_image, cv2.COLOR_RGB2BGR))

    return original_image, segmented_image