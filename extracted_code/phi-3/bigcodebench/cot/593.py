import csv
import os
from datetime import datetime, timedelta
from random import seed, randint
import matplotlib.pyplot as plt
import pandas as pd

seed(0)  # Ensure reproducibility

VEHICLE_TYPES = ['Car', 'Bus', 'Truck', 'Bike']
OUTPUT_DIR = './output'

def generate_traffic_data(hours, output_dir=OUTPUT_DIR):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=hours)
    time_range = [start_time + timedelta(minutes=i*15) for i in range((hours*60)//15)]

    data = []
    for time in time_range:
        row = {'Time': time.strftime('%H:%M')}
        for vehicle in VEHICLE_TYPES:
            row[vehicle] = randint(0, 100)
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, f'traffic_data_{hours}h.csv'), index=False)

    fig, ax = plt.subplots()
    for vehicle in VEHICLE_TYPES:
        ax.plot(df['Time'], df[vehicle], label=vehicle)
    ax.set_xlabel('Time')
    ax.set_ylabel('Vehicle Count')
    ax.set_title('Traffic Data')
    ax.legend()

    return os.path.join(output_dir, f'traffic_data_{hours}h.csv'), ax

generate_traffic_data(2)  # Generate data for 2 hours