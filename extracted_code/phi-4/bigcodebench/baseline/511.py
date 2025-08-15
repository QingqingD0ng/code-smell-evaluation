import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def task_func(column, data):
    if not data:
        stats = {'sum': 0,'mean': np.nan,'min': np.nan,'max': np.nan}
        fig, ax = plt.subplots()
        ax.pie([1], labels=["No Data"], startangle=90)
        ax.axis('equal')
        return stats, ax

    df = pd.DataFrame(data, columns=['Age', 'Salary', 'Experience'])

    if column not in df.columns:
        raise KeyError(f"Invalid column name: {column}")

    col_data = df[column]

    stats = {
       'sum': col_data.sum(),
       'mean': col_data.mean(),
       'min': col_data.min(),
       'max': col_data.max()
    }

    fig, ax = plt.subplots()
    ax.pie(col_data, labels=df['Age'], startangle=90)
    ax.axis('equal')

    return stats, ax