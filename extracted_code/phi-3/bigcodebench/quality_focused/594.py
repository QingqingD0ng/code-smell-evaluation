import csv
import os
import shutil
from datetime import datetime, timedelta
from random import choices

WEATHER_CONDITIONS = ['Sunny', 'Cloudy', 'Rainy', 'Snowy', 'Stormy']
OUTPUT_DIR = './output'
BACKUP_DIR = f'{OUTPUT_DIR}_backup'

def task_func(hours, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(BACKUP_DIR, exist_ok=True)
    
    base_time = datetime.now()
    time_format = "%Y-%m-%d %H:%M:%S"
    
    with open(os.path.join(output_dir, 'weather_data.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time', 'Condition'])
        
        for hour in range(hours):
            current_time = base_time + timedelta(hours=hour)
            condition = choices(WEATHER_CONDITIONS)[0]
            writer.writerow([current_time.strftime(time_format), condition])
    
    shutil.copy(os.path.join(output_dir, 'weather_data.csv'), BACKUP_DIR)
    return os.path.join(output_dir, 'weather_data.csv')

# Example usage:
# print('weather_data.csv' in task_func(24))
# print('weather_data.csv' in task_func(10))