import os
import csv
import random
from datetime import datetime

def task_func():
    directory = "task_func_data"
    file_path = os.path.join(directory, "Output.txt")

    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Prepare data
    data = [
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": round(random.uniform(15.0, 25.0), 2),
            "humidity": round(random.uniform(30.0, 50.0), 2)
        }
    ]

    # Write data to CSV file
    with open(file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["timestamp", "temperature", "humidity"])
        
        # Write header if file is empty
        if os.stat(file_path).st_size == 0:
            writer.writeheader()
        
        writer.writerow(data[0])

    # Return file path
    return file_path

# Clean up the file after use
def cleanup(file_path):
    os.remove(file_path)

# Example usage
file_path = task_func()
print(file_path)
cleanup(file_path)