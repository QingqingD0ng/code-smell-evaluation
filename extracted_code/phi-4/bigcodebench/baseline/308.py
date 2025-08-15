import pandas as pd
from statistics import mean
import random

FIELDS = ['Physics', 'Math', 'Chemistry', 'Biology', 'English', 'History']
STUDENTS = ['Student_' + str(i) for i in range(1, 101)]

def task_func(additional_fields=[]):
    fields = FIELDS + additional_fields
    data = {field: [random.randint(0, 100) for _ in STUDENTS] for field in fields}
    
    for student in STUDENTS:
        data[student +'Grade'] = [random.randint(0, 100) for _ in STUDENTS]
    
    df = pd.DataFrame(data, index=STUDENTS).T
    
    for student in STUDENTS:
        df.loc['Average', student +'Grade'] = mean(df.loc[student, fields])
    
    for field in fields:
        df.loc[field, 'Average Grade'] = mean(df.loc[field])
    
    df.loc['Average Grade', 'Average Grade'] = mean(df.loc['Average', fields])
    
    return df