import cv2
import numpy as np
import os
from sklearn.cluster import KMeans

def task_func(image_path='image.jpg', n_clusters=3, random_seed=42):
    if n_clusters < 1:
        raise ValueError("n_clusters must be a positive integer")
    if not os.path.isfile(image_path):
        raise FileNotFoundError("The image file does not exist at the specified path")

    # Read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixels = image.reshape((-1, 3))
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed)
    labels = kmeans.fit_predict(pixels)
    
    # Replace each pixel value with its nearby centroid
    segmented_image = kmeans.cluster_centers_[labels]
    segmented_image = segmented_image.astype(np.uint8)
    segmented_image = segmented_image.reshape(image.shape)
    
    # Return the original and segmented images as numpy arrays
    return image, segmented_image