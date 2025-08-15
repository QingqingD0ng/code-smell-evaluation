import csv
import os
from datetime import datetime, timedelta
from random import randint

SENSORS = ['Temperature', 'Humidity', 'Pressure']
OUTPUT_DIR = './output'
MIN_SENSOR_VALUE = 20
MAX_SENSOR_VALUE = 30

def generate_sensor_data(hours, sensor_value_range=(MIN_SENSOR_VALUE, MAX_SENSOR_VALUE), output_dir=OUTPUT_DIR):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file_path = f"{output_dir}/sensor_data_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    
    try:
        with open(output_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Time'] + SENSORS)
            
            start_time = datetime.now()
            for _ in range(hours * 60):
                current_time = start_time + timedelta(minutes=_)
                writer.writerow([current_time.strftime('%Y-%m-%d %H:%M:%S')] +
                                [randint(*sensor_value_range) for _ in SENSORS])
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")
    
    return output_file_path

# Example usage:
file_path = generate_sensor_data(1)
print(f"File created at: {file_path}")