import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


def task_func(data, n_components=2):

    if not isinstance(n_components, int) or n_components < 1:

        raise ValueError("n_components must be a positive integer")


    pca = PCA(n_components=n_components)

    transformed_data = pca.fit_transform(data)

    transformed_data_df = pd.DataFrame(transformed_data, columns=['PC1', 'PC2'])


    fig, ax = plt.subplots()

    scatter = ax.scatter(transformed_data_df['PC1'], transformed_data_df['PC2'])

    ax.set_xlabel('Principal Component 1')

    ax.set_ylabel('Principal Component 2')

    ax.set_title('PCA Scatter Plot')

    plt.legend(*scatter.legend_elements(), title="Legend")

    plt.show()


    return transformed_data_df, ax