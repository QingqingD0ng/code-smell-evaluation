import os
import csv
import random
from datetime import datetime

def task_func():
    os.makedirs('task_func_data', exist_ok=True)
    file_path = 'task_func_data/Output.txt'
    
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        for _ in range(5):  # Generate 5 random entries
            temperature = round(random.uniform(-10.0, 40.0), 2)
            humidity = round(random.uniform(20.0, 90.0), 2)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            writer.writerow([timestamp, temperature, humidity])
    
    result = file_path
    os.remove(file_path)
    
    return result