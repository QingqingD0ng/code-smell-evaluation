import pandas as pd

import numpy as np

import matplotlib.pyplot as plt


def task_func(column, data):

    if not data:

        return {'sum': 0,'mean': float('nan'),'min': float('nan'),'max': float('nan')}, None


    df = pd.DataFrame(data, columns=['Age', 'Salary', 'Experience'])


    if column not in df.columns:

        raise KeyError(f"Invalid column: {column}")


    stats = df[column].agg(['sum','mean','min','max']).to_dict()


    plt.figure()

    plt.pie(df[column], labels=df['Age'], autopct='%1.1f%%')

    ax = plt.gca()

    ax.set_title(f'Pie Chart of {column}')


    return stats, ax