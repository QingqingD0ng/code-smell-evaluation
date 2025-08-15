import os
import csv
import random
from datetime import datetime

def task_func():
    directory = "task_func_data"
    filename = "Output.txt"
    path = os.path.join(directory, filename)

    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Generate and write data
    with open(path, mode='a', newline='') as file:
        writer = csv.writer(file)
        for _ in range(5):  # Generate 5 rows of data
            temperature = random.uniform(-10.0, 40.0)  # Random temperature
            humidity = random.uniform(20.0, 100.0)  # Random humidity
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            writer.writerow([timestamp, temperature, humidity])

    # Return the path before deletion
    result = path

    # Delete the file after use
    os.remove(path)
    os.rmdir(directory)  # Remove directory if empty

    return result