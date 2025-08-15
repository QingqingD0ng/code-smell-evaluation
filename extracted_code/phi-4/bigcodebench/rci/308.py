import pandas as pd
from statistics import mean
import random

NUM_STUDENTS = 100
STUDENTS = ['Student_' + str(i) for i in range(1, NUM_STUDENTS + 1)]
FIELDS = ['Physics', 'Math', 'Chemistry', 'Biology', 'English', 'History']
AVERAGE_ROW_NAME = 'Average'
STUDENT_AVERAGE_COL_NAME = 'Student Average'

def generate_grades(num_students, subjects):
    return {subject: [random.randint(0, 100) for _ in range(num_students)] for subject in subjects}

def calculate_student_averages(df, subjects):
    for student in STUDENTS:
        student_grades = [df.loc[subject, student] for subject in subjects]
        df.loc[student, STUDENT_AVERAGE_COL_NAME] = mean(student_grades)

def calculate_subject_averages(df, subjects):
    for subject in subjects:
        df.loc[AVERAGE_ROW_NAME, subject] = mean(df.loc[subject, STUDENTS])
    df.loc[AVERAGE_ROW_NAME, STUDENT_AVERAGE_COL_NAME] = mean(df.loc[AVERAGE_ROW_NAME, subjects])

def task_func(additional_fields=[]):
    all_fields = FIELDS + additional_fields
    grades_data = generate_grades(NUM_STUDENTS, all_fields)
    
    for student in STUDENTS:
        grades_data[student + 'Grade'] = [random.randint(0, 100) for _ in STUDENTS]
    
    df = pd.DataFrame(grades_data, index=STUDENTS).T
    calculate_student_averages(df, all_fields)
    calculate_subject_averages(df, all_fields)
    
    return df