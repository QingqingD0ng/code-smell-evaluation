import csv
import os
from datetime import datetime, timedelta
from random import randint

SENSORS = ['Temperature', 'Humidity', 'Pressure']
OUTPUT_DIR = './output'

def task_func(hours, output_dir=OUTPUT_DIR):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    start_time = datetime.now()
    file_path = os.path.join(output_dir, f'sensor_data_{start_time.strftime("%Y%m%d_%H%M%S")}.csv')
    
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time', 'Temperature', 'Humidity', 'Pressure'])
        
        for hour in range(hours):
            for minute in range(60):
                current_time = start_time + timedelta(hours=hour, minutes=minute)
                temperature = randint(-10, 40)
                humidity = randint(20, 80)
                pressure = randint(950, 1050)
                
                writer.writerow([current_time.strftime('%Y-%m-%d %H:%M:%S'), temperature, humidity, pressure])
    
    return file_path

# Example usage
if __name__ == "__main__":
    file_path = task_func(1)
    print(f"File created at: {file_path}")