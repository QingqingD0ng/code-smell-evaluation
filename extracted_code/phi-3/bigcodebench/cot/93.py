import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def task_func(data, n_components=2):
    if not isinstance(n_components, int) or n_components <= 0:
        raise ValueError("n_components must be a positive integer")

    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data)
    transformed_data = pd.DataFrame(data=principal_components, columns=['PC' + str(i + 1) for i in range(n_components)])

    fig, ax = plt.subplots()
    ax.scatter(transformed_data['PC1'], transformed_data['PC2'])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Scatter plot of the transformed data')

    return transformed_data, ax