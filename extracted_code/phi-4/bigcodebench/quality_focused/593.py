import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from random import randint

VEHICLE_TYPES = ['Car', 'Bus', 'Truck', 'Bike']
OUTPUT_DIR = './output'


def task_func(hours, output_dir=OUTPUT_DIR):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamps = [datetime.now() + timedelta(hours=i) for i in range(hours)]
    data = {vehicle: [] for vehicle in VEHICLE_TYPES}
    data['Time'] = timestamps

    for _ in range(hours):
        for vehicle in VEHICLE_TYPES:
            data[vehicle].append(randint(0, 50))

    df = pd.DataFrame(data)
    file_path = os.path.join(output_dir, 'traffic_data.csv')
    df.to_csv(file_path, index=False)

    plt.figure(figsize=(10, 5))
    for vehicle in VEHICLE_TYPES:
        plt.plot(df['Time'], df[vehicle], label=vehicle)
    plt.xlabel('Time')
    plt.ylabel('Vehicle Count')
    plt.title('Traffic Data')
    plt.legend()
    plt.grid(True)

    plt.xticks(rotation=45)
    plt.tight_layout()
    ax = plt.gca()

    return file_path, ax