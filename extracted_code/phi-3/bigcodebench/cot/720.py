import os
import csv
import random
from datetime import datetime

def task_func():
    directory = "task_func_data"
    file_name = "Output.txt"
    file_path = os.path.join(directory, file_name)
    
    if not os.path.exists(directory):
        os.makedirs(directory)

    data_entries = 100
    sensor_data = [(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), random.uniform(15.0, 30.0), random.uniform(30.0, 60.0)) for _ in range(data_entries)]

    with open(file_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Temperature", "Humidity"])
        for entry in sensor_data:
            writer.writerow(entry)

    return file_path

# Example usage:
file_path = task_func()