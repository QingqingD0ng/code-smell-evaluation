import pandas as pd
import json
import os
import shutil

def task_func(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
    processed_path = os.path.join(path, "processed")
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
    
    all_files = sorted([f for f in os.listdir(path) if f.endswith('.json')])
    frames = []
    
    for file in all_files:
        file_path = os.path.join(path, file)
        with open(file_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df['source'] = file
        frames.append(df)
        
        shutil.move(file_path, os.path.join(processed_path, file))
    
    df = pd.concat(frames, ignore_index=True)
    return df