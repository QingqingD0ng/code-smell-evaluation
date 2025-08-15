import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def task_func(N=100, CATEGORIES=["A", "B", "C", "D", "E"], seed=42):
    np.random.seed(seed)

    # Ensure each category appears at least once if N >= len(CATEGORIES)
    if N >= len(CATEGORIES):
        categories = np.random.choice(CATEGORIES, N, replace=True)
    else:
        categories = np.random.choice(CATEGORIES, N, replace=False)
        np.random.shuffle(categories)

    # Create DataFrame
    df = pd.DataFrame({
        "x": np.random.rand(N),
        "y": np.random.rand(N),
        "category": categories
    })

    # Create scatter plot
    fig, ax = plt.subplots()
    scatter = ax.scatter(df["x"], df["y"], c=df["category"].astype("category").cat.codes)

    # Create legend
    handles, labels = scatter.legend_elements()
    ax.legend(handles, CATEGORIES, title="Category")

    return df, ax