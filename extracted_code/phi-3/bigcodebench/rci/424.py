import cv2
import numpy as np
import os
from sklearn.cluster import KMeans

def load_image(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"The image file {image_path} does not exist.")
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def initialize_kmeans(n_clusters, random_seed):
    return KMeans(n_clusters=n_clusters, random_state=random_seed)

def apply_kmeans(image, kmeans):
    pixels = image.reshape((-1, 3))
    labels = kmeans.fit_predict(pixels)
    segmented_image = kmeans.cluster_centers_[labels].astype(np.uint8)
    segmented_image = segmented_image.reshape(image.shape)
    return segmented_image

def task_func(image_path='image.jpg', n_clusters=3, random_seed=42):
    """
    Reads an RGB image, applies K-means clustering to segment the image into 'n_clusters' regions, 
    and saves each region as a separate image. The function returns numpy arrays of the original 
    and segmented images.
    """

    if not isinstance(n_clusters, int) or n_clusters < 1:
        raise ValueError("n_clusters must be a positive integer")

    image = load_image(image_path)
    kmeans = initialize_kmeans(n_clusters, random_seed)
    segmented_image = apply_kmeans(image, kmeans)

    return image, segmented_image