import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from datetime import datetime


def task_func(column, data):

    if column not in ['Date', 'Steps', 'Calories Burned', 'Distance Walked']:

        raise KeyError(f"Invalid column: {column}")


    if not data:

        raise ValueError("Data list is empty")


    df = pd.DataFrame(data, columns=['Date', column])


    for col in ['Steps', 'Calories Burned', 'Distance Walked']:

        if col not in df.columns:

            continue


        if (df[col] < 0).any():

            raise ValueError(f"Negative values found in column: {col}")


    stats = {

       'sum': df[column].sum(),

       'mean': df[column].mean(),

       'min': df[column].min(),

       'max': df[column].max()

    }


    ax = df.plot(x='Date', y=column, kind='line', title=f'Line Chart of {column}')


    return stats, ax