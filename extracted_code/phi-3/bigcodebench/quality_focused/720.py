import os

import csv

import random

from datetime import datetime


def task_func():
    file_path = "task_func_data/Output.txt"
    
    if not os.path.exists("task_func_data"):
        os.makedirs("task_func_data")
    
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        for _ in range(10):
            row = [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                random.uniform(20, 25),  # Temperature in Celsius
                random.uniform(40, 60)   # Humidity percentage
            ]
            writer.writerow(row)
    
    return file_path


# Usage

csv_path = task_func()

# The file will be deleted after its content is accessed