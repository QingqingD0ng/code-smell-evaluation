import os

import csv

import random

from statistics import mean


def generate_random_name():
    first_names = ['Alice', 'Bob', 'Charlie', 'Diana', 'Evan', 'Fiona', 'George', 'Hannah', 'Ian', 'Jane']
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez']
    return f"{random.choice(first_names)} {random.choice(last_names)}"


def task_func(filename):
    people_data = []
    for _ in range(PEOPLE_COUNT):
        name = generate_random_name()
        age = random.randint(18, 65)
        height = random.uniform(1.5, 2.0)
        weight = random.uniform(50, 100)
        people_data.append([name, age, round(height, 2), round(weight, 2)])
    avg_age, avg_height, avg_weight = mean(map(lambda x: x[1], people_data)), mean(map(lambda x: x[2], people_data)), mean(map(lambda x: x[3], people_data))
    people_data.append(['Average', f"{round(avg_age, 2)}", f"{round(avg_height, 2)}", f"{round(avg_weight, 2)}"])
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(COLUMNS)
        writer.writerows(people_data)
    return os.path.abspath(filename)


# Example usage

random.seed(0)
filename = 'people_report.csv'
path = task_func(filename)
print(os.path.exists(path))