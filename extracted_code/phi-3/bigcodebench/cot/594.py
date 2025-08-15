import csv
import os
import shutil
from datetime import datetime, timedelta
from random import choice

WEATHER_CONDITIONS = ['Sunny', 'Cloudy', 'Rainy', 'Snowy', 'Stormy']
OUTPUT_DIR = './output'
BACKUP_DIR = './backup'
FILE_NAME = 'weather_data.csv'

def create_backup_file_path(file_path):
    return os.path.join(BACKUP_DIR, os.path.basename(file_path))

def ensure_output_dir_exists(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def save_to_csv(file_path, data):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time', 'Condition'])
        writer.writerows(data)

def backup_file(file_path):
    backup_path = create_backup_file_path(file_path)
    shutil.copyfile(file_path, backup_path)

def generate_weather_data(hours, output_dir=OUTPUT_DIR):
    ensure_output_dir_exists(output_dir)
    start_time = datetime.now()
    data = [(start_time + timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S'), choice(WEATHER_CONDITIONS) for i in range(hours)]
    file_path = os.path.join(output_dir, FILE_NAME)
    save_to_csv(file_path, data)
    backup_file(file_path)
    return file_path

# Example usage:
print(generate_weather_data(24))