import os
import csv
import random
from statistics import mean

COLUMNS = ['Name', 'Age', 'Height', 'Weight']
PEOPLE_COUNT = 100

def task_func(filename):
    if not os.path.isfile(filename):
        with open(filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=COLUMNS)
            writer.writeheader()

            for _ in range(PEOPLE_COUNT):
                person = {
                    'Name': f'Person_{random.randint(1, PEOPLE_COUNT)}',
                    'Age': random.randint(18, 90),
                    'Height': random.uniform(1.5, 2.0),
                    'Weight': random.uniform(50, 100)
                }
                writer.writerow(person)

        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Average Age', mean([int(row['Age']) for row in csv.DictReader(open(filename))])])
            writer.writerow(['Average Height', mean([float(row['Height']) for row in csv.DictReader(open(filename))])])
            writer.writerow(['Average Weight', mean([float(row['Weight']) for row in csv.DictReader(open(filename))])])

    return os.path.abspath(filename)

# Example usage
# random.seed(0)
# filename = 'people_report.csv'
# path = task_func(filename)
# print(os.path.exists(path))  # Should print: True