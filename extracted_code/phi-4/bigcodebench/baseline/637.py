from random import sample
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def task_func(num_students: int) -> Tuple[pd.DataFrame, plt.Axes]:
    courses = ['Math', 'Science', 'History', 'English', 'Art']
    grades = {course: sample(range(0, 101), num_students) for course in courses}
    
    df = pd.DataFrame(grades)
    
    average_grades = df.mean()
    passing_counts = (df >= 60).sum()
    
    results = pd.DataFrame({'Average Grade': average_grades, 'Passing Counts': passing_counts})
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 2]})
    
    results.plot(kind='bar', ax=ax[1], title='Course-wise Average and Passing Grade Counts')
    
    ax[1].set_ylabel('Values')
    
    return df, ax[1]