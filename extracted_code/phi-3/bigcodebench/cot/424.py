import cv2
import numpy as np
import os
from sklearn.cluster import KMeans

def task_func(image_path='image.jpg', n_clusters=3, random_seed=42):
    if n_clusters < 1:
        raise ValueError('n_clusters must be a positive integer')

    if not os.path.exists(image_path):
        raise FileNotFoundError(f'The image file {image_path} does not exist')

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed)
    kmeans.fit(pixels)
    segmented_image = kmeans.cluster_centers_[kmeans.labels_]
    segmented_image = segmented_image.reshape(image.shape)
    return image, segmented_image

original_img_array, segmented_img_array = task_func('image.jpg', 3)
print(original_img_array.shape)
print(segmented_img_array.shape)