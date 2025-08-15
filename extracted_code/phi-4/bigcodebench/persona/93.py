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

    plt.figure(figsize=(8, 6))
    plt.scatter(transformed_df.iloc[:, 0], transformed_df.iloc[:, 1], edgecolor='k', s=50)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Scatter Plot')
    plt.grid(True)
    plt.axhline(0, color='grey', lw=0.5)
    plt.axvline(0, color='grey', lw=0.5)
    
    return transformed_df, plt.gca()

# Example usage:
# data = pd.DataFrame([[14, 25], [1, 22], [7, 8]], columns=['Column1', 'Column2'])
# transformed_data, plot = task_func(data)
# plt.show()  # To display the plot