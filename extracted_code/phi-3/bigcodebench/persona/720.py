import os

import csv

import random

from datetime import datetime


def task_func():
    file_path = "task_func_data/Output.txt"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Timestamp", "Temperature", "Humidity"])
        for _ in range(100):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            temperature = round(random.uniform(15.0, 30.0), 2)
            humidity = round(random.uniform(30.0, 90.0), 2)
            writer.writerow([timestamp, temperature, humidity])
    
    os.remove(file_path)
    
    return file_path


# To execute the function and print the file path before deletion

print(task_func())