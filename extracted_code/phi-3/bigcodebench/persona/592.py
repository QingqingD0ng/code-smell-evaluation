import csv
import os
from datetime import datetime, timedelta
from random import uniform

SENSORS = ['Temperature', 'Humidity', 'Pressure']
OUTPUT_DIR = './output'

def task_func(hours, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir,'sensor_data.csv')
    
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=hours)
    
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time', 'Temperature', 'Humidity', 'Pressure'])
        
        while start_time < end_time:
            current_time = start_time.strftime('%Y-%m-%d %H:%M:%S')
            data = [current_time] + [uniform(20, 30) for _ in SENSORS]  # Random values for sensor data
            writer.writerow(data)
            start_time += timedelta(minutes=1)
    
    return file_path