import csv
import os
from datetime import datetime, timedelta
from random import randint

SENSORS = ['Temperature', 'Humidity', 'Pressure']
OUTPUT_DIR = './output'

def task_func(hours, output_dir=OUTPUT_DIR):
    start_time = datetime.now()
    file_path = os.path.join(output_dir,'sensor_data.csv')

    os.makedirs(output_dir, exist_ok=True)

    with open(file_path, 'w', newline='') as csvfile:
        fieldnames = ['Time', 'Temperature', 'Humidity', 'Pressure']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for hour in range(hours):
            current_time = start_time + timedelta(hours=hour)
            for minute in range(60):
                data_time = current_time + timedelta(minutes=minute)
                row = {
                    'Time': data_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'Temperature': randint(-10, 40),
                    'Humidity': randint(20, 90),
                    'Pressure': randint(950, 1050)
                }
                writer.writerow(row)

    return file_path