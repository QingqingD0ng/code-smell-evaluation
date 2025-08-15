import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


def validate_data(data):

    if not isinstance(data, pd.DataFrame):

        raise TypeError("Input must be a pandas DataFrame")

    if not pd.api.types.is_numeric_dtype(data).all():

        raise ValueError("DataFrame must contain numeric data only")


def perform_pca(data, n_components):

    pca = PCA(n_components=n_components)

    return pca.fit_transform(data)


def plot_pca_scatter(data, ax):

    scatter = ax.scatter(data.iloc[:, 0], data.iloc[:, 1])

    ax.set_xlabel('Principal Component 1')

    ax.set_ylabel('Principal Component 2')

    ax.set_title('PCA Scatter Plot')

    legend = ax.legend(*scatter.legend_elements(), title="Legend")

    ax.add_artist(legend)


def task_func(data, n_components=2):

    validate_data(data)


    transformed_data = perform_pca(data, n_components)

    transformed_data_df = pd.DataFrame(transformed_data, columns=[f'PC{i+1}' for i in range(n_components)])


    fig, ax = plt.subplots()

    plot_pca_scatter(transformed_data_df, ax)


    plt.show()


# Example usage:

# data = pd.DataFrame([[14, 25], [1, 22], [7, 8]], columns=['Column1', 'Column2'])

# task_func(data)