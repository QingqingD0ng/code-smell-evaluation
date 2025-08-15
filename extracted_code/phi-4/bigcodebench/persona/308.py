import pandas as pd
from statistics import mean
import random

# Constants for generating the report data
FIELDS = ['Physics', 'Math', 'Chemistry', 'Biology', 'English', 'History']
STUDENTS = ['Student_' + str(i) for i in range(1, 101)]

def task_func(additional_fields=[]):
    # Combine the main fields with additional fields if any
    all_fields = FIELDS + additional_fields

    # Create a DataFrame with random grades for each student in each subject
    data = {field: [random.randint(0, 100) for _ in STUDENTS] for field in all_fields}
    df = pd.DataFrame(data, index=STUDENTS)

    # Calculate the average grade for each student and add it as a column
    df['Average Grade'] = df.mean(axis=1)

    # Calculate the average grade for each subject and add it as a row
    average_per_subject = pd.DataFrame(df.mean()).T
    average_per_subject.index = ['Average']

    # Concatenate the average grades per subject to the DataFrame
    final_report = pd.concat([df, average_per_subject])

    return final_report