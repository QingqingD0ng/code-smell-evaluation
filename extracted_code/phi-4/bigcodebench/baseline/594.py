import csv
import os
import shutil
from datetime import datetime
from random import randint

WEATHER_CONDITIONS = ['Sunny', 'Cloudy', 'Rainy', 'Snowy', 'Stormy']
OUTPUT_DIR = './output'
BACKUP_DIR = './backup'

def task_func(hours, output_dir=OUTPUT_DIR):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)

    filename = f'weather_data_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv'
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Time', 'Condition'])
        for hour in range(hours):
            time = (datetime.now() + timedelta(hours=hour)).strftime('%Y-%m-%d %H:%M:%S')
            condition = WEATHER_CONDITIONS[randint(0, len(WEATHER_CONDITIONS) - 1)]
            writer.writerow([time, condition])

    shutil.copy(filepath, BACKUP_DIR)
    return filepath