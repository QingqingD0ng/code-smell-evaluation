import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

def task_func(data_url="http://lib.stat.cmu.edu/datasets/boston", seed=42):
    try:
        np.random.seed(seed)
        data = pd.read_csv(data_url)
        correlation_matrix = data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=False)
        plt.title('Boston Housing Dataset Correlation Heatmap')
        plt.show()
        return plt.gca()
    except Exception as e:
        raise ValueError(f"An error occurred: {e}")