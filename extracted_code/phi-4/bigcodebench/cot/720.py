import os
import csv
import random
from datetime import datetime

def task_func():
    os.makedirs('task_func_data', exist_ok=True)
    file_path = 'task_func_data/Output.txt'
    
    temperature = round(random.uniform(15.0, 30.0), 2)
    humidity = round(random.uniform(30.0, 70.0), 2)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, temperature, humidity])
    
    result = file_path
    os.remove(file_path)
    
    return result