import csv

import os

import matplotlib.pyplot as plt

import pandas as pd

from datetime import datetime, timedelta


VEHICLE_TYPES = ['Car', 'Bus', 'Truck', 'Bike']
OUTPUT_DIR = './output'


def generate_traffic_data(hours):
    data = {'Time': [], 'Car': [], 'Bus': [], 'Truck': [], 'Bike': []}
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=hours)
    
    for hour in range(hours):
        time = start_time + timedelta(hours=hour)
        data['Time'].append(time)
        for vehicle_type in VEHICLE_TYPES:
            traffic_count = randint(50, 500)
            data[vehicle_type].append(traffic_count)
    
    return data


def save_data_to_csv(data, file_path):
    try:
        pd.DataFrame(data).to_csv(file_path, index=False)
    except IOError as e:
        print(f"An error occurred while writing to {file_path}: {e}")
        raise


def plot_traffic_data(df, ax):
    for vehicle_type in df.columns[1:]:
        ax.plot(df['Time'], df[vehicle_type], label=vehicle_type)
    ax.set_xlabel('Time')
    ax.set_ylabel('Vehicle Count')
    ax.legend()
    plt.xticks(rotation=45)


def task_func(hours, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    data = generate_traffic_data(hours)
    file_path = os.path.join(output_dir, 'traffic_data.csv')
    save_data_to_csv(data, file_path)
    
    fig, ax = plt.sub