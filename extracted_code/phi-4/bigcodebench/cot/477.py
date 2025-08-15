import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def task_func(N=100, CATEGORIES=["A", "B", "C", "D", "E"], seed=42):
    np.random.seed(seed)
    
    if N >= len(CATEGORIES):
        categories = np.random.choice(CATEGORIES, N - len(CATEGORIES))
        categories = np.concatenate((categories, CATEGORIES))
    else:
        categories = np.random.choice(CATEGORIES, N, replace=False)
    
    np.random.shuffle(categories)
    
    x = np.random.rand(N)
    y = np.random.rand(N)
    
    df = pd.DataFrame({'x': x, 'y': y, 'category': categories})
    
    fig, ax = plt.subplots()
    scatter = ax.scatter(df['x'], df['y'], c=df['category'].astype('category').cat.codes)
    legend1 = ax.legend(*scatter.legend_elements(), title="Categories")
    ax.add_artist(legend1)
    
    return df, ax