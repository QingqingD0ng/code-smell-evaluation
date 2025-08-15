import os
import csv
import random
from statistics import mean

COLUMNS = ['Name', 'Age', 'Height', 'Weight']
PEOPLE_COUNT = 100

def generate_name():
    first_names = ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank']
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Miller']
    return f"{random.choice(first_names)} {random.choice(last_names)}"

def generate_person():
    name = generate_name()
    age = random.randint(18, 99)
    height = round(random.uniform(150, 200), 2)
    weight = round(random.uniform(50, 100), 2)
    return [name, age, height, weight]

def task_func(filename):
    people_data = [generate_person() for _ in range(PEOPLE_COUNT)]

    ages = [person[1] for person in people_data]
    heights = [person[2] for person in people_data]
    weights = [person[3] for person in people_data]

    averages = [mean(ages), mean(heights), mean(weights)]
    people_data.append(['Average'] + averages)

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(COLUMNS + ['Average'])
        writer.writerows(people_data)

    return os.path.abspath(filename)

# Example usage
random.seed(0)
filename = 'people_report.csv'
path = task_func(filename)
print(os.path.exists(path))