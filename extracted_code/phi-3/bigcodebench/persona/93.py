import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


def task_func(data, n_components=2):
    if not isinstance(n_components, int) or n_components <= 0:
        raise ValueError("n_components must be a positive integer")

    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data)

    df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

    fig, ax = plt.subplots()
    ax.scatter(df_pca['PC1'], df_pca['PC2'])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Scatter plot of PCA transformed data')

    return df_pca, ax