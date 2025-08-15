from random import sample
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def task_func(num_students: int) -> Tuple[pd.DataFrame, plt.Axes]:
    # Define courses
    courses = ['Math', 'Science', 'History', 'Art']
    
    # Generate random grades for each student in each course
    data = {course: sample(range(0, 101), num_students) for course in courses}
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Calculate average grade for each course
    averages = df.mean()
    
    # Calculate number of students with passing grade (>= 60) for each course
    passing_counts = (df >= 60).sum()
    
    # Prepare data for plotting
    num_courses = len(courses)
    x = np.arange(num_courses)
    width = 0.35
    
    # Create bar plot
    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, averages, width, label='Average Grade')
    bars2 = ax.bar(x + width/2, passing_counts, width, label='Passing Count')
    
    # Add some text for labels, title, and custom x-axis tick labels
    ax.set_xlabel('Courses')
    ax.set_ylabel('Scores')
    ax.set_title('Course-wise Average and Passing Grade Counts')
    ax.set_xticks(x)
    ax.set_xticklabels(courses)
    ax.legend()
    
    # Function to attach a text label above each bar
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(bars1)
    autolabel(bars2)
    
    return df, ax