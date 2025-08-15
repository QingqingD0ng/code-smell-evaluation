import os
import csv
import random
from statistics import mean

def generate_random_data(count):
    names = [f'Person{idx}' for idx in range(count)]
    ages = [random.randint(18, 70) for _ in range(count)]
    heights = [random.uniform(1.5, 2.0) for _ in range(count)]
    weights = [random.uniform(50, 100) for _ in range(count)]
    return list(zip(names, ages, heights, weights))

def calculate_averages(data, keys):
    return {key: round(mean(map(lambda x: x[idx] for x in data)), 2) for idx, key in enumerate(keys)}

def task_func(filename):
    data = generate_random_data(PEOPLE_COUNT)
    avg_data = calculate_averages(data, COLUMNS)
    avg_data['Average'] = [avg_data['Age'], avg_data['Height'], avg_data['Weight']]
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(COLUMNS + ['Average'])
        writer.writerows(data + [avg_data['Average']])
    
    return os.path.abspath(filename)