import pandas as pd
import random
from statistics import mean

FIELDS = ['Physics', 'Math', 'Chemistry', 'Biology', 'English', 'History']
STUDENTS = [f'Student_{i}' for i in range(1, 101)]

def generate_grades(additional_fields):
    grades = {field: [random.randint(0, 100) for _ in STUDENTS] for field in FIELDS + additional_fields}
    return grades

def calculate_averages(grades):
    averages = {student: mean(grades[subject][student] for subject in grades) for student in STUDENTS}
    return averages

def create_report(additional_fields=[]):
    grades = generate_grades(additional_fields)
    student_averages = calculate_averages(grades)
    df = pd.DataFrame(grades, index=STUDENTS)
    df.loc['Average Grade'] = student_averages.values()
    df.loc['Average Grade'] = df.loc['Average Grade'].round(2)
    return df

# Example usage:
# random.seed(0)
# report = create_report(['Computer Science', 'Geography'])
# print(report.columns)