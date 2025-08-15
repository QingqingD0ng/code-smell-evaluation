import csv
import os
import shutil
from datetime import datetime, timedelta
from random import choice

WEATHER_CONDITIONS = ['Sunny', 'Cloudy', 'Rainy', 'Snowy', 'Stormy']
OUTPUT_DIR = './output'
BACKUP_DIR = './backup'

def task_func(hours, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(BACKUP_DIR, exist_ok=True)

    file_path = os.path.join(output_dir, 'weather_data.csv')
    backup_path = os.path.join(BACKUP_DIR, f'weather_data_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv')

    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time', 'Condition'])

        for hour in range(hours):
            current_time = (datetime.now() + timedelta(hours=hour)).strftime('%Y-%m-%d %H:%M:%S')
            condition = choice(WEATHER_CONDITIONS)
            writer.writerow([current_time, condition])

    shutil.copy(file_path, backup_path)
    return file_path