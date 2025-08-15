import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

sns.set(font_scale=1.2)

def task_func(data_url="http://lib.stat.cmu.edu/datasets/boston", seed=42):
    np.random.seed(seed)
    try:
        data = pd.read_csv(data_url)
    except Exception as e:
        raise ValueError("Failed to load data") from e
    
    plt.figure(figsize=(10, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Boston Housing Correlation Heatmap")
    plt.savefig("boston_housing_correlation_heatmap.png")
    plt.close()
    return plt.gca()

if __name__ == "__main__":
    ax = task_func()
    print(type(ax))