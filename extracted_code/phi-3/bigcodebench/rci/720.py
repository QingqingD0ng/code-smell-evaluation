import os
import csv
import random
from datetime import datetime
from pathlib import Path

def generate_sensor_data():
    return [[datetime.now().strftime("%Y-%m-%d %H:%M:%S"), random.uniform(20, 30), random.uniform(30, 60)] for _ in range(100)]

def task_func():
    file_path = Path("task_func_data/Output.txt")
    os.makedirs(file_path.parent, exist_ok=True)

    try:
        with file_path.open(mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "temperature", "humidity"])
            data = generate_sensor_data()
            writer.writerows(data)
        return str(file_path)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    file_path = task_func()
    if file_path:
        print(f"CSV file path: {file_path}")