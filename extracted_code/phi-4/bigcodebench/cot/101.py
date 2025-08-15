import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

def task_func(data_url="http://lib.stat.cmu.edu/datasets/boston", seed=42, save_path=None):
    np.random.seed(seed)

    try:
        data = pd.read_csv(data_url, sep="\s+", header=None, skiprows=22)
    except Exception as e:
        raise ValueError("Error loading data") from e

    columns = [
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", 
        "PTRATIO", "B", "LSTAT", "MEDV"
    ]
    data.columns = columns

    corr = data.corr()

    plt.rcParams['font.family'] ='sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'

    try:
        ax = plt.figure(figsize=(10, 8)).gca()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, square=True, cbar_kws={"shrink":.5})
        plt.title("Correlation Heatmap of Boston Housing Dataset")
    except Exception as e:
        raise ValueError("Error generating heatmap") from e

    if save_path:
        try:
            plt.savefig(save_path)
        except Exception as e:
            raise ValueError("Error saving the plot") from e

    return ax