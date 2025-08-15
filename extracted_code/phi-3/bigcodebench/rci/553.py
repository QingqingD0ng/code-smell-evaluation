import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COLUMN_NAMES = ['A', 'B', 'C', 'D', 'E']

def create_and_plot_dataframe(row_indices: list, column_labels: list) -> plt.Axes:
    """
    Generates a DataFrame with random values and plots it as a bar chart.

    Parameters:
    - row_indices (list): List of row indices for the DataFrame.
    - column_labels (list): List of column names for the DataFrame.

    Returns:
    - matplotlib.axes.Axes: The Axes object of the plotted bar chart.
    """
    dataframe = pd.DataFrame(np.random.rand(len(row_indices), len(column_labels)), index=row_indices, columns=column_labels)
    axes = dataframe.plot(kind='bar')
    return axes

axes_object = create_and_plot_dataframe([1, 2, 3], ['A', 'B', 'C', 'D', 'E'])
plt.show()