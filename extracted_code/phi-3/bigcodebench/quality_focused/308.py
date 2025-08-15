import pandas as pd
from statistics import mean
import random

# Constants for generating the report data
FIELDS = ['Physics', 'Math', 'Chemistry', 'Biology', 'English', 'History']
STUDENTS = ['Student_{}'.format(i) for i in range(1, 101)]

def task_func(additional_fields=[]):
    all_fields = FIELDS + additional_fields
    data = {field: [random.randint(0, 100) for _ in STUDENTS] for field in all_fields}
    df = pd.DataFrame(data, index=STUDENTS)
    df['Average Grade'] = df.mean(axis=1)
    df.loc['Average Grade'] = df.mean()
    return df