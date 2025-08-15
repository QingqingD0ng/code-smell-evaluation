import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def task_func(N=100, CATEGORIES=["A", "B", "C", "D", "E"], seed=42):
    np.random.seed(seed)
    
    x = np.random.rand(N)
    y = np.random.rand(N)

    if N >= len(CATEGORIES):
        categories = np.random.choice(CATEGORIES, N, replace=True)
    else:
        categories = np.random.choice(CATEGORIES, N, replace=False)
        categories = np.concatenate([categories, np.random.choice(CATEGORIES, N - len(CATEGORIES), replace=True)])
        np.random.shuffle(categories)

    df = pd.DataFrame({'x': x, 'y': y, 'category': categories})

    fig, ax = plt.subplots()
    scatter = ax.scatter(df['x'], df['y'], c=df['category'].astype('category').cat.codes)
    legend1 = ax.legend(*scatter.legend_elements(), title="Categories")
    ax.add_artist(legend1)

    return df, ax