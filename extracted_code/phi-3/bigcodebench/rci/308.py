import pandas as pd
from statistics import mean
import random

MAX_GRADE = 100
ROUNDING_PRECISION = 2

def generate_students(num_students):
    return [f"Student_{i}" for i in range(1, num_students + 1)]

def generate_fields(base_fields, additional_fields):
    all_fields = base_fields + additional_fields
    return all_fields

def task_func(num_students, additional_fields=[]):
    if not all(isinstance(field, str) for field in additional_fields):
        raise ValueError("All additional fields must be strings.")
    if len(additional_fields)!= len(set(additional_fields)):
        raise ValueError("Additional fields must be unique.")
    
    students = generate_students(num_students)
    all_fields = generate_fields(['Physics', 'Math', 'Chemistry', 'Biology', 'English', 'History'], additional_fields)
    
    data = {field: [round(random.uniform(0, MAX_GRADE), ROUNDING_PRECISION) for _ in students] for field in all_fields}
    df = pd.DataFrame(data, index=students)
    df.loc['Average Grade'] = df.mean()
    
    return df

# Example usage:
# random.seed(0)
# report = task_func(100, ['Computer Science', 'Geography'])
# print(report.columns)
This version includes modifications based on the code smells identified:

- Replaced magic numbers with named constants.
- Dynamically generated students and fields using list comprehensions and functions.
- Added input validation for additional fields.
- Applied rounding to the random grade generation.

These changes should make the code more maintainable, robust, and readable.