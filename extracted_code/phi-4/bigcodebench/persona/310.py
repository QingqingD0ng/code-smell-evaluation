import os
import csv
import random
from statistics import mean

COLUMNS = ['Name', 'Age', 'Height', 'Weight']
PEOPLE_COUNT = 100

def task_func(filename):
    random.seed(0)  # Ensure reproducibility
    names = [f"Person_{i}" for i in range(1, PEOPLE_COUNT + 1)]
    
    ages = [random.randint(18, 75) for _ in range(PEOPLE_COUNT)]
    heights = [random.uniform(150, 200) for _ in range(PEOPLE_COUNT)]
    weights = [random.uniform(50, 100) for _ in range(PEOPLE_COUNT)]

    data = list(zip(names, ages, heights, weights))

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(COLUMNS)
        writer.writerows(data)

        avg_age = mean(ages)
        avg_height = mean(heights)
        avg_weight = mean(weights)

        writer.writerow(['Average', avg_age, avg_height, avg_weight])

    return os.path.abspath(filename)