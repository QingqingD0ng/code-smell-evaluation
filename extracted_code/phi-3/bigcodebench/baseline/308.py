import pandas as pd
from statistics import mean
import random

FIELDS = ['Physics', 'Math', 'Chemistry', 'Biology', 'English', 'History']
STUDENTS = ['Student_' + str(i) for i in range(1, 101)]

def task_func(additional_fields=[]):
    all_fields = FIELDS + additional_fields
    data = {field: [random.randint(0, 100) for _ in STUDENTS] for field in all_fields}
    df = pd.DataFrame(data, index=STUDENTS)
    df.loc['Average Grade'] = df.mean()
    df.loc['Average Grade'] = df.loc['Average Grade'].round(2)
    return df