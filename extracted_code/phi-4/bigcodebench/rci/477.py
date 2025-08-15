import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_N = 100
DEFAULT_CATEGORIES = ["A", "B", "C", "D", "E"]
DEFAULT_SEED = 42

def task_func(N=DEFAULT_N, CATEGORIES=DEFAULT_CATEGORIES, seed=DEFAULT_SEED):
    np.random.seed(seed)
    
    x = np.random.rand(N)
    y = np.random.rand(N)

    if N >= len(CATEGORIES):
        categories = np.random.choice(CATEGORIES, N, replace=True)
    else:
        categories = np.random.choice(CATEGORIES, len(CATEGORIES), replace=False)
        additional_categories = np.random.choice(CATEGORIES, N - len(CATEGORIES), replace=True)
        categories = np.concatenate([categories, additional_categories])
        np.random.shuffle(categories)

    df = pd.DataFrame({'x': x, 'y': y, 'category': categories})

    fig, ax = plt.subplots()
    scatter = ax.scatter(df['x'], df['y'], c=df['category'].astype('category').cat.codes, cmap='viridis')
    legend1 = ax.legend(*scatter.legend_elements(), title="Categories", loc="upper right")
    ax.add_artist(legend1)

    return df, ax