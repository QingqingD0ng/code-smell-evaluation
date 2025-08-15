import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

def task_func(data_url="http://lib.stat.cmu.edu/datasets/boston", seed=42, save_path=None):
    np.random.seed(seed)

    try:
        data = pd.read_csv(data_url, sep="\s+", header=None)
        data.columns = [
            "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX",
            "PTRATIO", "B", "LSTAT", "MEDV"
        ]

        corr = data.corr()

        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(
            corr, 
            annot=True, 
            fmt=".2f", 
            cmap="coolwarm", 
            cbar_kws={"shrink": 0.8},
            xticklabels=corr.columns,
            yticklabels=corr.columns
        )

        plt.xticks(fontsize=10, fontname='Arial')
        plt.yticks(fontsize=10, fontname='Arial')
        plt.xlabel('Variables', fontsize=12, fontname='Arial')
        plt.ylabel('Variables', fontsize=12, fontname='Arial')

        if save_path:
            plt.savefig(save_path, dpi=300)

        return ax

    except Exception as e:
        raise ValueError("An error occurred in generating or saving the plot.") from e