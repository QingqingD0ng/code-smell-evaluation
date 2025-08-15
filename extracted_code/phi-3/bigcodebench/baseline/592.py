import csv
import os
from datetime import datetime, timedelta
from random import randint

SENSORS = ['Temperature', 'Humidity', 'Pressure']
OUTPUT_DIR = './output'

def task_func(hours, output_dir=OUTPUT_DIR):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = f"{OUTPUT_DIR}/sensor_data_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time'] + SENSORS)
        start_time = datetime.now()
        for _ in range(hours * 60):
            current_time = start_time + timedelta(minutes=_)
            writer.writerow([current_time.strftime('%Y-%m-%d %H:%M:%S')] +
                            [randint(20, 30) for _ in SENSORS])
    return filename.replace(OUTPUT_DIR, '')