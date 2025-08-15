from random import sample
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def task_func(num_students: int) -> Tuple[pd.DataFrame, plt.Axes]:
    courses = ['Math', 'Science', 'History', 'English']
    grades = {course: sample(range(50, 101), num_students) for course in courses}
    df = pd.DataFrame(grades)
    df.index = sample(range(1, num_students + 1), num_students)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Student ID'}, inplace=True)

    averages = df.mean()
    passing_counts = (df >= 60).sum()

    fig, ax = plt.subplots()
    averages.plot(kind='bar', yerr=df.std(), ax=ax, capsize=4)
    ax.set_title('Course-wise Average and Passing Grade Counts')
    ax.set_ylabel('Average Grade')
    passing_counts.plot(kind='bar', ax=ax, color='red', secondary_y=True)
    ax.right_ax.set_ylabel('Passing Students')
    plt.tight_layout()

    return df, ax