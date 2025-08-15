import os
import csv
import random
from statistics import mean

COLUMNS = ['Name', 'Age', 'Height', 'Weight']
PEOPLE_COUNT = 100

def task_func(filename):
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(COLUMNS)
            data = []
            for _ in range(PEOPLE_COUNT):
                name = f'Person_{random.randint(1, 1000)}'
                age = random.randint(18, 75)
                height = random.uniform(150.0, 200.0)
                weight = random.uniform(50.0, 100.0)
                row = [name, age, height, weight]
                writer.writerow(row)
                data.append(row)
            avg_age = mean(row[1] for row in data)
            avg_height = mean(row[2] for row in data)
            avg_weight = mean(row[3] for row in data)
            writer.writerow(['Average', avg_age, avg_height, avg_weight])
    return os.path.abspath(filename)