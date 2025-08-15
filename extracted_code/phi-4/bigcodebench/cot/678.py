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

    all_data = []
    
    for filename in sorted(os.listdir(path)):
        file_path = os.path.join(path, filename)
        
        if os.path.isfile(file_path) and filename.endswith('.json'):
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data)
            df['source'] = filename
            
            all_data.append(df)
            
            shutil.move(file_path, os.path.join(processed_path, filename))
    
    final_df = pd.concat(all_data, ignore_index=True)
    
    return final_df