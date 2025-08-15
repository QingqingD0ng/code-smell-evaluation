from random import sample
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def task_func(num_students: int) -> Tuple[pd.DataFrame, plt.Axes]:
    courses = ['Math', 'Science', 'History', 'Art']
    grades = {course: sample(range(0, 101), num_students) for course in courses}
    df = pd.DataFrame(grades)

    averages = df.mean().rename('Average')
    passing_counts = (df >= 60).sum().rename('Passing Counts')

    summary_df = pd.concat([averages, passing_counts], axis=1)

    fig, ax = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [1, 3]})
    ax[0].axis('off')
    ax[0].table(cellText=summary_df.values, colLabels=summary_df.columns, rowLabels=summary_df.index, loc='center')
    ax[0].set_title('Course-wise Average and Passing Grade Counts')

    ax[1].bar(summary_df.index, summary_df['Average'], label='Average', alpha=0.6, color='b')
    ax[1].bar(summary_df.index, summary_df['Passing Counts'], label='Passing Counts', alpha=0.6, color='r', bottom=summary_df['Average'])
    ax[1].set_ylabel('Counts and Grades')
    ax[1].set_title('Course-wise Average and Passing Grade Counts')
    ax[1].legend()

    plt.tight_layout()
    return df, ax[1]