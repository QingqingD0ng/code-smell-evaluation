import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def task_func(num_students: int) -> Tuple[pd.DataFrame, plt.Axes]:

    courses = ['Math', 'Science', 'History', 'English', 'Art']

    grades = {course: np.random.randint(80 if course == 'Math' else 60, 100, num_students // 5) for course in courses}

    df = pd.DataFrame(grades).reset_index().melt(id_vars=['index', 'level_0'], var_name='Course', value_name='Grade')

    df['Grade'] = df['Grade'].astype(int)

    grade_intervals = pd.cut(df['Grade'], bins=[0, 59, 69, 79, 100], labels=['F', 'D', 'C', 'B'])

    df['Grade_Interval'] = grade_intervals

    df_agg = df.groupby(['Course', 'Grade_Interval']).size().reset_index(name='Count')

    avg_grades = df.groupby('Course')['Grade'].mean().reset_index(name='Average')

    merged = pd.merge(avg_grades, df_agg, on='Course')

    fig, ax = plt.subplots()

    merged.plot(x='Course', y='Count', kind='bar', legend=False, ax=ax, color='skyblue')

    ax.set_title('Course-wise Average and Passing Grade Counts')

    return df, ax

# Example usage:

# df, ax = task_func(50)

# print(ax.get_title())