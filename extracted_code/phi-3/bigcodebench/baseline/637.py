import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def task_func(num_students):
    courses = ['Math', 'Science', 'History', 'English', 'Art']
    grades = {course: {'A': np.random.randint(80, 100, num_students // 5) if course!= 'Math' else np.random.randint(60, 100, num_students // 5).tolist(),
                      'B': np.random.randint(50, 79, num_students // 5),
                      'C': np.random.randint(0, 49, num_students // 5),
                      'D': np.random.randint(0, 19, num_students // 5)}
               for course in courses}
    
    df = pd.DataFrame(grades)
    df = df.stack().reset_index()
    df.columns = ['Student', 'Course', 'Grade']
    df['Grade'] = df['Grade'].astype(int)
    
    avg_grades = df.groupby('Course')['Grade'].mean().reset_index()
    passing = df[df['Grade'] >= 60].groupby('Course')['Grade'].count().reset_index()
    
    merged = pd.merge(avg_grades, passing, on='Course')
    
    fig, ax = plt.subplots()
    merged.plot(x='Course', y='Grade', kind='bar', legend=False, ax=ax, color='skyblue')
    merged.plot(x='Course', y='Grade', kind='bar', legend=False, ax=ax, color='lightblue')
    ax.set_title('Course-wise Average and Passing Grade Counts')
    
    return df, ax

# Example usage:
# df, ax = task_func(50)
# print(ax.get_title())