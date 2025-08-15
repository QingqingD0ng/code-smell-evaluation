import pandas as pd
import json
import os
import shutil

def task_func(path):
    full_path = os.path.abspath(path)

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    json_files = [f.name for f in os.scandir(full_path) if f.is_file() and f.name.endswith('.json')]
    json_files.sort()

    df_list = []
    for json_file in json_files:
        file_path = os.path.join(full_path, json_file)
        with open(file_path, 'r') as f:
            data = json.load(f)
            data_df = pd.DataFrame(data)
            data_df['source'] = json_file
            df_list.append(data_df)

    df = pd.concat(df_list, ignore_index=True)

    processed_dir = os.path.join(full_path, 'processed')
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    for json_file in json_files:
        shutil.move(os.path.join(full_path, json_file), os.path.join(processed_dir, json_file))

    return df