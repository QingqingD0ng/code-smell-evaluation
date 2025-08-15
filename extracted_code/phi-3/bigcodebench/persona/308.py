import pandas as pd

from statistics import mean

import random


FIELDS = ['Physics', 'Math', 'Chemistry', 'Biology', 'English', 'History']

STUDENTS = ['Student_{}'.format(i) for i in range(1, 101)]

additional_fields = []


def generate_grades():

    grades = {student: {field: random.randint(0, 100) for field in FIELDS + additional_fields} for student in STUDENTS}

    return grades


def calculate_averages(grades):

    averages = {field: mean([grades[student][field] for student in grades]) for field in FIELDS}

    return averages


def task_func(additional_fields):

    global additional_fields

    additional_fields = additional_fields

    grades = generate_grades()

    averages = calculate_averages(grades)


    df = pd.DataFrame(grades).T

    df.loc['Average Grade'] = df.mean()

    df.loc['Average Grade'].name = 'Average Grade'

    df.columns.name = 'Subjects'

    df.insert(0, 'Average Grade', df.mean(axis=1))

    return df


# Example usage:

random.seed(0)

report = task_func(['Computer Science', 'Geography'])

print(report.columns)