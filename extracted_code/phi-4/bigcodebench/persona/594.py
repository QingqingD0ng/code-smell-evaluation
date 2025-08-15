import csv
import os
import shutil
from datetime import datetime
from random import choice

WEATHER_CONDITIONS = ['Sunny', 'Cloudy', 'Rainy', 'Snowy', 'Stormy']
OUTPUT_DIR = './output'
BACKUP_DIR = './backup'

def task_func(hours, output_dir=OUTPUT_DIR):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    csv_file_path = os.path.join(output_dir, 'weather_data.csv')
    
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time', 'Condition'])
        
        for hour in range(hours):
            current_time = (datetime.now() + timedelta(hours=hour)).strftime('%Y-%m-%d %H:%M:%S')
            condition = choice(WEATHER_CONDITIONS)
            writer.writerow([current_time, condition])
    
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
    
    backup_file_path = os.path.join(BACKUP_DIR, f'backup_{datetime.now().strftime("%Y%m%d%H%M%S")}_weather_data.csv')
    shutil.copyfile(csv_file_path, backup_file_path)
    
    return csv_file_path