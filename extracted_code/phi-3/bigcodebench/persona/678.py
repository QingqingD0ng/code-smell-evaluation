import pandas as pd
import json
import os
import shutil

def task_func(path):
    if not os.path.exists(path):
        os.makedirs(path)
    json_files = sorted([f for f in os.listdir(path) if f.endswith('.json')])
    df_list = []
    for json_file in json_files:
        with open(os.path.join(path, json_file), 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df['source'] = json_file
        df_list.append(df)
    processed_path = os.path.join(path, 'processed')
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
    for df, json_file in zip(df_list, json_files):
        df.to_csv(os.path.join(processed_path, json_file.replace('.json', '.csv')), index=False)
    return pd.concat(df_list, ignore_index=True)