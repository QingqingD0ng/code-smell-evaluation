import os
import csv
import random
from statistics import mean

COLUMNS = ['Name', 'Age', 'Height', 'Weight']
PEOPLE_COUNT = 100
NAMES = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Heidi", "Ivan", "Judy"]

def task_func(filename):
    random.seed(0)
    data = []

    for _ in range(PEOPLE_COUNT):
        name = random.choice(NAMES)
        age = random.randint(18, 70)
        height = round(random.uniform(150, 200), 2)
        weight = round(random.uniform(50, 100), 2)
        data.append([name, age, height, weight])

    averages = [mean(column) for column in zip(*data)[1:]]

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(COLUMNS)
        writer.writerows(data)
        writer.writerow(['Average'] + averages)

    return os.path.abspath(filename)