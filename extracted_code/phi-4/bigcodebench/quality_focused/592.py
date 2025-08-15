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
        writer = csv.DictWriter(file, fieldnames=['Time', 'Temperature', 'Humidity', 'Pressure'])
        writer.writeheader()
        
        start_time = datetime.now()
        for _ in range(hours * 60):  # Assuming data is generated per minute
            current_time = start_time + timedelta(minutes=_)
            row = {
                'Time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'Temperature': randint(-10, 40),
                'Humidity': randint(20, 90),
                'Pressure': randint(950, 1050)
            }
            writer.writerow(row)
    
    return file_path