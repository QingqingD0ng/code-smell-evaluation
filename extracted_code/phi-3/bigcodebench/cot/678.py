import pandas as pd
import json
import os
import shutil

def task_func(path):
    if not os.path.exists(path):
        os.makedirs(path)
    json_files = sorted([f for f in os.listdir(path) if f.endswith('.json')])
    all_data = []
    for filename in json_files:
        file_path = os.path.join(path, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)
            for item in data:
                item['Source'] = filename
                all_data.append(item)
    df = pd.DataFrame(all_data)
    processed_path = os.path.join(path, 'processed')
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
    for filename in json_files:
        shutil.move(os.path.join(path, filename), os.path.join(processed_path, filename))
    return df