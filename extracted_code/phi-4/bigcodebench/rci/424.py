import cv2
import numpy as np
import os
from sklearn.cluster import KMeans

DEFAULT_RANDOM_SEED = 42
IMAGE_EXTENSION = '.jpg'

def save_cluster_image(cluster_image, cluster_index, output_dir='.'):
    filename = os.path.join(output_dir, f'cluster_{cluster_index}{IMAGE_EXTENSION}')
    cv2.imwrite(filename, cv2.cvtColor(cluster_image, cv2.COLOR_RGB2BGR))

def task_func(image_path='image.jpg', n_clusters=3, random_seed=DEFAULT_RANDOM_SEED, output_dir='.'):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File {image_path} does not exist.")

    if not isinstance(n_clusters, int) or n_clusters < 1:
        raise ValueError("'n_clusters' must be a positive integer.")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at {image_path} could not be read.")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image.copy()

    height, width, depth = image.shape
    image_reshaped = image.reshape(-1, depth)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed)
    kmeans.fit(image_reshaped)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    segmented_image = centers[labels].reshape(height, width, depth).astype(np.uint8)

    for i in range(n_clusters):
        cluster_image = np.zeros_like(image)
        cluster_image[labels == i] = centers[i]
        save_cluster_image(cluster_image, i, output_dir)

    return original_image, segmented_image