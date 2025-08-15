import csv
import os
import shutil
from datetime import datetime, timedelta
from random import choice

WEATHER_CONDITIONS = ['Sunny', 'Cloudy', 'Rainy', 'Snowy', 'Stormy']
OUTPUT_DIR = './output'

def task_func(hours, output_dir=OUTPUT_DIR):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now()
    file_name = f"weather_data_{timestamp.strftime('%Y%m%d%H%M%S')}.csv"
    file_path = os.path.join(output_dir, file_name)
    
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time', 'Condition'])
        for hour in range(hours):
            time = (timestamp + timedelta(hours=hour)).strftime('%Y-%m-%d %H:%M:%S')
            condition = choice(WEATHER_CONDITIONS)
            writer.writerow([time, condition])
    
    backup_dir = os.path.join(output_dir, 'backup')
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    shutil.copy(file_path, backup_dir)
    
    return file_path

# Example usage:
# print('weather_data.csv' in task_func(24))
# print('weather_data.csv' in task_func(10))