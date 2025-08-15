import csv
import os
import shutil
from datetime import datetime, timedelta
from random import choice

WEATHER_CONDITIONS = ['Sunny', 'Cloudy', 'Rainy', 'Snowy', 'Stormy']
OUTPUT_DIR = './output'
BACKUP_DIR = os.path.join(OUTPUT_DIR, 'backup')

def task_func(hours, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(BACKUP_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'weather_data_{timestamp}.csv'
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Time', 'Condition'])
        writer.writeheader()

        current_time = datetime.now()
        for _ in range(hours):
            condition = choice(WEATHER_CONDITIONS)
            writer.writerow({'Time': current_time.strftime("%Y-%m-%d %H:%M:%S"), 'Condition': condition})
            current_time += timedelta(hours=1)

    shutil.copy(filepath, BACKUP_DIR)
    return filepath