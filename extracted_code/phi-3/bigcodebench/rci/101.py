import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

def load_data(data_url="http://lib.stat.cmu.edu/datasets/boston", seed=42):
    np.random.seed(seed)
    try:
        return pd.read_csv(data_url)
    except Exception as error:
        raise IOError("Failed to load data") from error

def generate_correlation_heatmap(data, file_name="boston_housing_correlation_heatmap.png"):
    plt.figure(figsize=(10, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Boston Housing Correlation Heatmap")
    
    try:
        plt.savefig(file_name)
    except Exception as error:
        raise IOError("Failed to save plot") from error
    
    plt.close()

def task_func(data_url="http://lib.stat.cmu.edu/datasets/boston", seed=42, file_name="boston_housing_correlation_heatmap.png"):
    data = load_data(data_url, seed)
    ax = generate_correlation_heatmap(data, file_name)
    return ax

if __name__ == "__main__":
    ax = task_func()
    print(type(ax))