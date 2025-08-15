import os
import csv
import random
from statistics import mean

COLUMNS = ['Name', 'Age', 'Height', 'Weight']
PEOPLE_COUNT = 100

def task_func(filename):
    data = [
        [f"Person {i}", random.randint(18, 65), random.uniform(1.5, 2.0), random.uniform(50, 100)]
        for i in range(PEOPLE_COUNT)
    ]
    averages = {
        'Average Age': mean(row[1] for row in data),
        'Average Height': mean(row[2] for row in data),
        'Average Weight': mean(row[3] for row in data)
    }
    data.append([f"Averages", *averages.values()])
    
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(COLUMNS + list(averages.keys()))
        writer.writerows(data)
    
    return os.path.abspath(filename)