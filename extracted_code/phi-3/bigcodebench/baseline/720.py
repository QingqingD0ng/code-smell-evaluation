import os
import csv
import random
from datetime import datetime

def task_func():
    file_path = "task_func_data/Output.txt"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    header = ["timestamp", "temperature", "humidity"]
    data = [
        [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            random.uniform(20, 30),  # Random temperature between 20 and 30
            random.uniform(30, 60),  # Random humidity between 30% and 60%
        ]
        for _ in range(100)  # Generate 100 rows of data
    ]

    with open(file_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)

    return file_path

# Example of usage:
if __name__ == "__main__":
    file_path = task_func()
    print(f"CSV file path: {file_path}")