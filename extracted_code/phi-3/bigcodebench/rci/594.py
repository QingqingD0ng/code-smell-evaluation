import csv
import os
import shutil
from datetime import datetime, timedelta
from random import choice
from typing import Any

WEATHER_CONDITIONS = ['Sunny', 'Cloudy', 'Rainy', 'Snowy', 'Stormy']
OUTPUT_DIR = './output'
BACKUP_DIR = os.path.join(OUTPUT_DIR, 'backup')

def task_func(hours: int, output_dir: str = OUTPUT_DIR) -> str:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now()
    file_name = f"weather_data_{timestamp.strftime('%Y%m%d%H%M%S')}.csv"
    file_path = os.path.join(output_dir, file_name)

    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time', 'Condition'])

        for hour in range(hours):
            time = (datetime.now() + timedelta(hours=hour)).strftime('%Y-%m-%d %H:%M:%S')
            condition = choice(WEATHER_CONDITIONS)
            writer.writerow([time, condition])

    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR, exist_ok=True)

    shutil.copy(file_path, BACKUP_DIR)

    return file_path

# Example usage:
# print('weather_data.csv' in task_func(24))
# print('weather_data.csv' in task_func(10))