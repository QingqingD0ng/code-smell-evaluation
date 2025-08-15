import pandas as pd
import json
import os
import shutil

def task_func(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
    processed_path = os.path.join(path, 'processed')
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)

    data_frames = []
    files = sorted([f for f in os.listdir(path) if f.endswith('.json')])

    for file in files:
        with open(os.path.join(path, file), 'r') as f:
            data = json.load(f)
            df = pd.DataFrame(data)
            df['source'] = file
            data_frames.append(df)
        
        shutil.move(os.path.join(path, file), os.path.join(processed_path, file))
    
    result_df = pd.concat(data_frames, ignore_index=True)
    return result_df