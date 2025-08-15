import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

def task_func(data_url="http://lib.stat.cmu.edu/datasets/boston", seed=42, save_path=None):
    np.random.seed(seed)
    
    data = pd.read_csv(data_url, sep="\s+", header=None)
    data.columns = [
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", 
        "PTRATIO", "B", "LSTAT", "MEDV"
    ]
    
    corr = data.corr()
    
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", 
                     xticklabels=corr.columns, yticklabels=corr.columns)
    
    plt.xticks(fontname="Arial")
    plt.yticks(fontname="Arial")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")
        except Exception as e:
            raise ValueError(f"Error saving plot: {e}")
    
    return ax