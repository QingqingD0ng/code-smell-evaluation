import csv

import os

from datetime import datetime, timedelta

from random import randint

import matplotlib.pyplot as plt

import pandas as pd


def task_func(hours, output_dir=OUTPUT_DIR):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=hours)
    time_slots = pd.date_range(start=start_time, end=end_time, freq='H')
    
    data = []
    for time in time_slots:
        row = [time.strftime('%Y-%m-%d %H:%M:%S'),
               randint(100, 1000), randint(50, 500), randint(20, 200), randint(5, 50)]
        data.append(row)
    
    df = pd.DataFrame(data, columns=['Time', 'Car', 'Bus', 'Truck', 'Bike'])
    file_path = os.path.join(output_dir, 'traffic_data.csv')
    df.to_csv(file_path, index=False)
    
    ax = df.plot(x='Time', kind='line', title='Traffic Data')
    
    plt.xlabel('Time')
    plt.ylabel('Vehicle Count')
    
    return file_path, ax