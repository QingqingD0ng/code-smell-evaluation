import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


def task_func(data, n_components=2):
    if not isinstance(n_components, int) or n_components <= 0:
        raise ValueError("n_components must be a positive integer")

    pca = PCA(n_components=n_components)
    transformed_data = pd.DataFrame(pca.fit_transform(data), columns=['PC1', 'PC2'] + [f'PC{i+3}' for i in range(n_components-2)])
    
    plt.figure()
    plt.scatter(transformed_data['PC1'], transformed_data['PC2'])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Scatter Plot')
    plt.grid(True)
    
    axes = plt.gca()
    return transformed_data, axes