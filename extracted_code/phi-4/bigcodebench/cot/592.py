import csv
import os
from datetime import datetime, timedelta
from random import randint

SENSORS = ['Temperature', 'Humidity', 'Pressure']
OUTPUT_DIR = './output'

def task_func(hours, output_dir=OUTPUT_DIR):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir,'sensor_data.csv')
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time', *SENSORS])

        start_time = datetime.now()
        for _ in range(hours * 60):
            current_time = start_time + timedelta(minutes=_)
            row = [
                current_time.strftime('%Y-%m-%d %H:%M:%S'),
                randint(-20, 50),
                randint(0, 100),
                randint(900, 1100)
            ]
            writer.writerow(row)

    return file_path