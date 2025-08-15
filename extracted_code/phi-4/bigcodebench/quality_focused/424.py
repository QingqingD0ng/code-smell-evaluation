import cv2
import numpy as np
import os
from sklearn.cluster import KMeans

def task_func(image_path='image.jpg', n_clusters=3, random_seed=42):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at {image_path}")
    if n_clusters < 1:
        raise ValueError("n_clusters must be a positive integer")

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_img_array = img.copy()

    img_reshaped = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed)
    labels = kmeans.fit_predict(img_reshaped)
    centroids = kmeans.cluster_centers_

    segmented_img_reshaped = centroids[labels].astype('uint8')
    segmented_img_array = segmented_img_reshaped.reshape(img.shape)

    output_dir = "segmented_images"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(n_clusters):
        cluster_img = np.zeros_like(img)
        cluster_img[labels == i] = centroids[i]
        cluster_img = cluster_img.astype('uint8')
        cluster_img = cv2.cvtColor(cluster_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, f"segment_{i}.jpg"), cluster_img)

    return original_img_array, segmented_img_array