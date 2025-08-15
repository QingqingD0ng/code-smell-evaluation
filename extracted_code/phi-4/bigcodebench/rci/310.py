import os
import csv
import random
from statistics import mean

COLUMNS = ['Name', 'Age', 'Height', 'Weight']
PEOPLE_COUNT = 100
MIN_AGE, MAX_AGE = 18, 75
MIN_HEIGHT, MAX_HEIGHT = 150.0, 200.0
MIN_WEIGHT, MAX_WEIGHT = 50.0, 100.0

def generate_person_data():
    name = f'Person_{random.randint(1, 1000)}'
    age = random.randint(MIN_AGE, MAX_AGE)
    height = random.uniform(MIN_HEIGHT, MAX_HEIGHT)
    weight = random.uniform(MIN_WEIGHT, MAX_WEIGHT)
    return [name, age, height, weight]

def calculate_averages(data):
    avg_age = mean(row[1] for row in data)
    avg_height = mean(row[2] for row in data)
    avg_weight = mean(row[3] for row in data)
    return ['Average', avg_age, avg_height, avg_weight]

def task_func(filename):
    if not isinstance(filename, str):
        raise ValueError("Filename must be a string.")
    
    data = [generate_person_data() for _ in range(PEOPLE_COUNT)]
    averages = calculate_averages(data)
    
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(COLUMNS)
        writer.writerows(data)
        writer.writerow(averages)
    
    return os.path.abspath(filename)