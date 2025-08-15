import os
import csv
import random
from statistics import mean

COLUMNS = ['Name', 'Age', 'Height', 'Weight']
PEOPLE_COUNT = 100

def task_func(filename):
    data = []
    names = [f'Person{i}' for i in range(PEOPLE_COUNT)]
    ages = [random.randint(18, 70) for _ in range(PEOPEN_COUNT)]
    heights = [random.uniform(1.5, 2.0) for _ in range(PEOPLE_COUNT)]
    weights = [random.uniform(50, 100) for _ in range(PEOPLE_COUNT)]
    
    for name, age, height, weight in zip(names, ages, heights, weights):
        data.append([name, age, height, weight])
    
    avg_age = round(mean(ages), 2)
    avg_height = round(mean(heights), 2)
    avg_weight = round(mean(weights), 2)
    
    data.append(['Average', avg_age, avg_height, avg_weight])
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(COLUMNS)
        writer.writerows(data)
    
    return os.path.abspath(filename)