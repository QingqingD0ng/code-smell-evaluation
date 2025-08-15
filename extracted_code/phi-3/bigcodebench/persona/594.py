import csv
import os
import shutil
from datetime import datetime, timedelta
from random import choice

def task_func(hours, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    filename = f'weather_data_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv'
    filepath = os.path.join(output_dir, filename)
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time', 'Condition'])
        for hour in range(hours):
            current_time = datetime.now() + timedelta(hours=hour)
            writer.writerow([current_time.strftime('%Y-%m-%d %H:%M:%S'), choice(WEATHER_CONDITIONS)])
    backup_filename = f'{filename}.bak'
    shutil.move(filepath, os.path.join(output_dir, backup_filename))
    return filepath

# Example usage:
task_func(24)
task_func(10)