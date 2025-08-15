import pandas as pd
import json
import os
import shutil

def task_func(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
    if not os.path.exists(os.path.join(path, 'processed')):
        os.makedirs(os.path.join(path, 'processed'))
    
    all_data = []
    
    for filename in sorted(os.listdir(path)):
        if filename.endswith('.json'):
            file_path = os.path.join(path, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
                df = pd.DataFrame(data)
                df['Source'] = filename
                all_data.append(df)
            
            shutil.move(file_path, os.path.join(path, 'processed', filename))
    
    df = pd.concat(all_data, ignore_index=True)
    return df