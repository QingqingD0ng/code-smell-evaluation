import csv
import os
from datetime import datetime, timedelta
from random import randint
import matplotlib.pyplot as plt
import pandas as pd

VEHICLE_TYPES = ['Car', 'Bus', 'Truck', 'Bike']
OUTPUT_DIR = './output'

def task_func(hours, output_dir=OUTPUT_DIR):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    start_time = datetime.now()
    data = {'Time': [], 'Car': [], 'Bus': [], 'Truck': [], 'Bike': []}
    
    for i in range(hours):
        current_time = start_time + timedelta(hours=i)
        data['Time'].append(current_time.strftime('%Y-%m-%d %H:%M:%S'))
        for vehicle in VEHICLE_TYPES:
            data[vehicle].append(randint(0, 100))
    
    csv_file_path = os.path.join(output_dir, 'traffic_data.csv')
    
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        writer.writeheader()
        writer.writerows([dict(zip(data.keys(), values)) for values in zip(*data.values())])
    
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    ax = df.set_index('Time').plot(kind='line')
    ax.set_xlabel('Time')
    ax.set_ylabel('Vehicle Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return csv_file_path, ax