import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def task_func(num_students: int) -> Tuple[pd.DataFrame, plt.Axes]:
    courses = ['Math', 'Science', 'History', 'Art', 'Physical Education']
    grades = np.random.randint(0, 101, size=(num_students, len(courses)))
    df = pd.DataFrame(grades, columns=courses)
    df['Passed'] = df.apply(lambda row: row.gt(60).sum(), axis=1)
    df['Average'] = df.mean(axis=1)
    
    ax = plt.figure().add_subplot(111)
    df.plot(kind='bar', y=['Average', 'Passed'], ax=ax)
    ax.set_title('Course-wise Average and Passing Grade Counts')
    return df, ax

# Example usage:
df, ax = task_func(50)
plt.show()