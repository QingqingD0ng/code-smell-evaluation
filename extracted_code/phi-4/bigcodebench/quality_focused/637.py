from random import sample
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def task_func(num_students: int) -> Tuple[pd.DataFrame, plt.Axes]:
    courses = ['Math', 'Science', 'History', 'Art']
    grades = {course: sample(range(50, 101), num_students) for course in courses}
    
    df = pd.DataFrame(grades)
    averages = df.mean()
    passing_counts = (df >= 60).sum()

    data = pd.DataFrame({'Average': averages, 'Passing Count': passing_counts})
    ax = data.plot(kind='bar', subplots=True, figsize=(10, 6), layout=(1, 2), sharey=True)
    ax[0].set_title('Average Grades')
    ax[1].set_title('Passing Grade Counts')
    fig = ax[0].get_figure()
    fig.suptitle('Course-wise Average and Passing Grade Counts')

    return df, ax[0]

# Example usage:
# df, ax = task_func(50)
# ax.get_figure().show()