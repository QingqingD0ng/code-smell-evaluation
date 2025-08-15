import csv

import os

from datetime import datetime, timedelta

import matplotlib.pyplot as plt

import pandas as pd


VEHICLE_TYPES = ['Car', 'Bus', 'Truck', 'Bike']
OUTPUT_DIR = './output'


def task_func(hours, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    data = {'Time': [], 'Car': [], 'Bus': [], 'Truck': [], 'Bike': []}
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=hours)
    
    for hour in range(hours):
        time = start_time + timedelta(hours=hour)
        data['Time'].append(time)
        for vehicle_type in VEHICLE_TYPES:
            traffic_count = randint(50, 500)
            data[vehicle_type].append(traffic_count)
    
    df = pd.DataFrame(data)
    file_path = os.path.join(output_dir, 'traffic_data.csv')
    df.to_csv(file_path, index=False)
    
    fig, ax = plt.subplots()
    for vehicle_type in VEHICLE_TYPES:
        ax.plot(df['Time'], df[vehicle_type], label=vehicle_type)
    ax.set_xlabel('Time')
    ax.set_ylabel('Vehicle Count')
    ax.legend()
    plt.xticks(rotation=45)
    
    return file_path, ax


# Example usage

file_path, ax = task_func(2)