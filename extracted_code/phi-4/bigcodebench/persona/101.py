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
    
    plt.style.use('seaborn-whitegrid')
    plt.rcParams['font.sans-serif'] = ['Arial']
    
    corr = data.corr()
    ax = sns.heatmap(
        corr, 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm", 
        cbar_kws={"shrink":.5}
    )
    
    if save_path:
        try:
            plt.savefig(save_path, format='png', dpi=300)
        except Exception as e:
            raise ValueError(f"Error saving plot: {e}")
    
    plt.show()
    return ax