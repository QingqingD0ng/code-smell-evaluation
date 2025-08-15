import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def task_func(column, data):
    if not data:
        return {'sum': 0,'mean': np.nan,'min': np.nan,'max': np.nan}, None

    df = pd.DataFrame(data, columns=['Age', 'Salary', 'Experience'])
    if column not in df.columns:
        raise KeyError(f"Invalid column: {column}")

    stats = {
      'sum': df[column].sum(),
      'mean': df[column].mean(),
      'min': df[column].min(),
      'max': df[column].max()
    }

    fig, ax = plt.subplots()
    df[column].plot(kind='pie', autopct='%1.1f%%', labels=df[column].astype(str), ax=ax)
    ax.axis('equal')
    plt.show()

    return stats, ax

# Example usage:
data = [[25, 50000, 2], [30, 75000, 5], [35, 100000, 7], [40, 125000, 10], [45, 150000, 12]]
stats, ax = task_func('Salary', data)
print(stats)