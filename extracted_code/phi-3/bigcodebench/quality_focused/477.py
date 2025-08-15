import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def task_func(N=100, CATEGORIES=["A", "B", "C", "D", "E"], seed=42):
    np.random.seed(seed)
    df = pd.DataFrame({
        "x": np.random.rand(N),
        "y": np.random.rand(N),
        "category": np.random.choice(CATEGORIES, size=N, replace=True)
    })

    if N < len(CATEGORIES):
        df["category"] = np.random.choice(CATEGORIES, size=N, replace=False)

    ax = df.plot.scatter(x="x", y="y", c="category", colormap="viridis")

    return df, ax

# Example usage:
df, ax = task_func()
print(df.head())