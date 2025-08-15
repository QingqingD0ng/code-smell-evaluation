import os
import csv
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from random import randint
from typing import Tuple

VEHICLE_TYPES = ['Car', 'Bus', 'Truck', 'Bike']
OUTPUT_DIR = './output'

def task_func(hours: int, output_dir: str = OUTPUT_DIR) -> Tuple[str, plt.Axes]:
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"traffic_data_{hours}hrs.csv"
    file_path = os.path.join(output_dir, file_name)
    
    time_range = pd.date_range(start=datetime.datetime.now(), periods=hours, freq='H')
    data = {
        'Time': time_range,
        'Car': [randint(0, 100) for _ in range(hours)],
        'Bus': [randint(0, 20) for _ in range(hours)],
        'Truck': [randint(0, 15) for _ in range(hours)],
        'Bike': [randint(0, 50) for _ in range(hours)]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    
    ax = df.set_index('Time').plot(kind='line')
    return file_path, ax

file_path, ax = task_func(2)
assert 'traffic_data_2hrs.csv' in file_path
assert isinstance(ax, plt.Axes)