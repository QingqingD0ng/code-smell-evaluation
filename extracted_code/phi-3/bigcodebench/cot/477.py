import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def task_func(N=100, CATEGORIES=["A", "B", "C", "D", "E"], seed=42):
    np.random.seed(seed)
    categories = np.random.choice(CATEGORIES, size=N, replace=True)
    x = np.random.rand(N)
    y = np.random.rand(N)
    category_counts = {category: np.sum(categories == category) for category in CATEGORIES}
    df = pd.DataFrame({'x': x, 'y': y, 'category': categories})
    
    for category in CATEGORIES:
        if category_counts[category] < 1:
            while category_counts[category] < 1:
                extra_index = np.random.choice(np.where(categories == category)[0])
                df.iloc[extra_index] = [df.iloc[extra_index]['x'], df.iloc[extra_index]['y'], category]
                category_counts[category] += 1
    
    df = df.sort_values('category').reset_index(drop=True)
    
    fig, ax = plt.subplots()
    for category in CATEGORIES:
        category_data = df[df['category'] == category]
        ax.scatter(category_data['x'], category_data['y'], label=category)
    ax.legend()
    
    return df, ax

# Example usage
df, ax = task_func()
plt.show()