import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def task_func(N=100, CATEGORIES=["A", "B", "C", "D", "E"], seed=42):
    np.random.seed(seed)
    
    # Ensure each category appears at least once if N >= number of categories
    if N >= len(CATEGORIES):
        categories = np.random.choice(CATEGORIES, size=N-len(CATEGORIES), replace=True).tolist()
        categories = CATEGORIES + categories
        np.random.shuffle(categories)
    else:
        categories = np.random.choice(CATEGORIES, size=N, replace=False).tolist()
    
    # Generate random x and y values
    x = np.random.rand(N)
    y = np.random.rand(N)
    
    # Create DataFrame
    df = pd.DataFrame({'x': x, 'y': y, 'category': categories})
    
    # Plotting
    fig, ax = plt.subplots()
    scatter = ax.scatter(df['x'], df['y'], c=df['category'].astype('category').cat.codes)
    legend1 = ax.legend(*scatter.legend_elements(), title="Categories")
    ax.add_artist(legend1)
    
    return df, ax