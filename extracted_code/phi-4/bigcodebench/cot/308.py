import pandas as pd
import random
from statistics import mean

def task_func(additional_fields=[]):
    subjects = ['Physics', 'Math', 'Chemistry', 'Biology', 'English', 'History'] + additional_fields
    
    data = {subject: [random.randint(0, 100) for _ in range(100)] for subject in subjects}
    data['Average Grade'] = [mean(data[subject][i] for subject in subjects) for i in range(100)]
    
    df = pd.DataFrame(data, index=['Student_' + str(i) for i in range(1, 101)])
    df.loc['Average'] = df.mean()
    
    return df