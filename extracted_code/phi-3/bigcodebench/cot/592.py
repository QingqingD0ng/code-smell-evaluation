import csv
import os
from datetime import datetime, timedelta
from random import uniform

SENSORS = ['Temperature', 'Humidity', 'Pressure']
OUTPUT_DIR = './output'

def task_func(hours, output_dir=OUTPUT_DIR):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = f"sensor_data_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time', 'Temperature', 'Humidity', 'Pressure'])
        start_time = datetime.now()
        for _ in range(hours * 60):  # Assuming we want data for each minute
            current_time = start_time + timedelta(minutes=_+1)
            row = [current_time.strftime('%Y-%m-%d %H:%M:%S'),
                   round(uniform(10.0, 30.0), 2),  # Random temperature
                   round(uniform(30.0, 70.0), 2),  # Random humidity
                   round(uniform(980.0, 1030.0), 2)  # Random pressure
                   ]
            writer.writerow(row)
    return hours