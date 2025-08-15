import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def task_func(N=100, CATEGORIES=["A", "B", "C", "D", "E"], seed=42):
    np.random.seed(seed)
    if N < len(CATEGORIES):
        categories = np.random.choice(CATEGORIES, size=N, replace=False)
    else:
        categories = np.random.choice(CATEGORIES, size=N, replace=True)
    x = np.random.rand(N)
    y = np.random.rand(N)
    df = pd.DataFrame({"x": x, "y": y, "category": categories})
    fig, ax = plt.subplots()
    for category in CATEGORIES:
        subset = df[df["category"] == category]
        ax.scatter(subset["x"], subset["y"], label=category)
    ax.legend()
    return df, ax