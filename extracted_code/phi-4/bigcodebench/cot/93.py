import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def task_func(data, n_components=2):
    if not isinstance(n_components, int) or n_components <= 0:
        raise ValueError("n_components must be a positive integer")

    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(data)
    transformed_df = pd.DataFrame(transformed_data, columns=[f'PC{i+1}' for i in range(n_components)])

    fig, ax = plt.subplots()
    ax.scatter(transformed_df.iloc[:, 0], transformed_df.iloc[:, 1] if n_components > 1 else np.zeros(transformed_df.shape[0]))
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2' if n_components > 1 else '')

    return transformed_df, ax