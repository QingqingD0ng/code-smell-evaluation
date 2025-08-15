import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def task_func(N=100, CATEGORIES=["A", "B", "C", "D", "E"], seed=42):
    np.random.seed(seed)
    categories = np.random.choice(CATEGORIES, size=N, replace=True)
    x = np.random.rand(N)
    y = np.random.rand(N)
    df = pd.DataFrame({"x": x, "y": y, "category": categories})
    fig, ax = plt.subplots()
    scatter_data = {category: df[df["category"] == category] for category in CATEGORIES}
    for category, subset in scatter_data.items():
        ax.scatter(subset["x"], subset["y"], label=category)
    ax.set_title('Scatter Plot of Categories')
    ax.set_xlabel('X values')
    ax.set_ylabel('Y values')
    ax.legend()
    return df, ax