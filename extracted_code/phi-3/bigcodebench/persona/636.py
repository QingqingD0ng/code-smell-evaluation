import matplotlib.pyplot as plt

import numpy as np

import pandas as pd


def task_func(rows):

    # Generate DataFrame with random integers

    df = pd.DataFrame(np.random.randint(0, 10, size=(rows, len(COLUMNS))), columns=COLUMNS)


    # Count non-zero values in each column

    non_zero_counts = df.notnull().sum()


    # Plotting bar plot

    fig, ax = plt.subplots()

    ax.bar(non_zero_counts.index, non_zero_counts.values)

    ax.set_title('Non-Zero Value Counts')

    return df, ax