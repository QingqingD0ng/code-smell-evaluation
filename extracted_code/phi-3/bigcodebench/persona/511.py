import pandas as pd

import numpy as np

import matplotlib.pyplot as plt


def task_func(column, data):

    df = pd.DataFrame(data, columns=['Age', 'Salary', 'Experience'])

    if column not in df.columns:

        raise KeyError(f"Invalid column: {column}")


    stats = {

       'sum': df[column].sum(),

       'mean': df[column].mean(),

       'min': df[column].min(),

       'max': df[column].max(),

    }


    if df[column].empty:

        stats['mean'] = np.nan

        stats['min'] = np.nan

        stats['max'] = np.nan


    fig, ax = plt.subplots()

    ax.pie(df[column], labels=df['Age'], autopct='%1.1f%%', startangle=90)

    plt.show()


    return stats, ax