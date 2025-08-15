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
    data = []
    
    for hour in range(hours):
        time = (start_time + timedelta(hours=hour)).strftime('%Y-%m-%d %H:%M:%S')
        row = {'Time': time}
        for vehicle in VEHICLE_TYPES:
            row[vehicle] = randint(0, 100)
        data.append(row)
    
    file_path = os.path.join(output_dir, 'traffic_data.csv')
    
    with open(file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Time'] + VEHICLE_TYPES)
        writer.writeheader()
        writer.writerows(data)
    
    df = pd.read_csv(file_path)
    df.set_index('Time', inplace=True)
    
    ax = df.plot(kind='line')
    ax.set_xlabel('Time')
    ax.set_ylabel('Vehicle Count')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return file_path, ax