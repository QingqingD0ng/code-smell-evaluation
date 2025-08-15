import random

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

from typing import Tuple


def task_func(num_students: int) -> Tuple[pd.DataFrame, plt.Axes]:

    # Generate random grades for 100 students in 5 courses

    courses = ['Math', 'Science', 'History', 'Art', 'Music']

    grades = {course: [random.randint(0, 100) for _ in range(num_students)] for course in courses}


    # Create a DataFrame from the grades dictionary

    df = pd.DataFrame(grades)


    # Calculate the average grade in each course

    average_grades = df.mean()


    # Determine the number of students with a passing grade (>= 60)

    passing_counts = (df >= 60).sum()


    # Combine the average grades and passing counts into a single DataFrame

    result_df = pd.DataFrame({

        'Average Grade': average_grades,

        'Passing Count': passing_counts

    })


    # Visualize the information using a bar plot

    fig, ax = plt.subplots()

    result_df.plot(kind='bar', ax=ax)

    ax.set_title('Course-wise Average and Passing Grade Counts')

    ax.set_xlabel('Course')

    ax.set_ylabel('Grades')

    plt.show()


    return result_df, ax