import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def task_func(N=100, CATEGORIES=["A", "B", "C", "D", "E"], seed=42):
    np.random.seed(seed)
    n_categories = len(CATEGORIES)
    min_samples_per_category = N // n_categories + 1 if N % n_categories!= 0 else N // n_categories
    
    categories_df = pd.DataFrame({
        "category": np.random.choice(CATEGORIES, size=min_samples_per_category, replace=False),
        "x": np.random.rand(min_samples_per_category),
        "y": np.random.rand(min_samples_per_category)
    })
    
    additional_rows = N - min_samples_per_category * n_categories
    if additional_rows > 0:
        additional_df = pd.DataFrame({
            "category": np.random.choice(CATEGORIES, size=additional_rows, replace=True),
            "x": np.random.rand(additional_rows),
            "y": np.random.rand(additional_rows)
        })
        categories_df = pd.concat([categories_df, additional_df], ignore_index=True)
    
    df = categories_df.sample(frac=1).reset_index(drop=True)
    ax = plt.scatter(df["x"], df["y"], c=df["category"].astype('category').cat.codes, cmap='viridis')
    plt.colorbar(ax, label='Category')
    return df, ax

# Example usage
df, ax = task_func()
df.head()
type(ax)