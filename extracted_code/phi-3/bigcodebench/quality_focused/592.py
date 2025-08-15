import csv

import os

from datetime import datetime, timedelta

from random import uniform


SENSORS = ['Temperature', 'Humidity', 'Pressure']

OUTPUT_DIR = './output'


def generate_sensor_data(hours, output_dir=OUTPUT_DIR):
    timestamp = datetime.now()
    data = []

    for hour in range(hours):
        current_time = timestamp + timedelta(hours=hour)
        row = {
            'Time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'Temperature': round(uniform(15, 25), 2),
            'Humidity': round(uniform(30, 60), 2),
            'Pressure': round(uniform(980, 1050), 2)
        }
        data.append(row)
        timestamp += timedelta(hours=1)

    file_path = os.path.join(output_dir,'sensor_data.csv')
    os.makedirs(output_dir, exist_ok=True)
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Time', 'Temperature', 'Humidity', 'Pressure'])
        writer.writeheader()
        writer.writerows(data)

    return file_path


# Example usage:

# file_path = generate_sensor_data(1)

# print(os.path.exists(file_path))

# print(isinstance(file_path, str))

# print('sensor_data.csv' in file_path)