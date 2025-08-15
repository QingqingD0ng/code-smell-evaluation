import pandas as pd
from statistics import mean
import random

FIELDS = ['Physics', 'Math', 'Chemistry', 'Biology', 'English', 'History']

def task_func(additional_fields=[]):
    fields = FIELDS + additional_fields
    students = ['Student_' + str(i) for i in range(1, 101)]
    data = {student: [random.randint(0, 100) for _ in fields] for student in students}

    df = pd.DataFrame(data, index=fields).T
    df['Average Grade'] = df.mean(axis=1)

    subject_averages = df.mean().to_frame().T
    subject_averages.index = ['Average']
    report_df = pd.concat([df, subject_averages])

    return report_df