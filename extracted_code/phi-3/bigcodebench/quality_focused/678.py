import pandas as pd
import json
import os
import shutil

def task_func(path):
    if not os.path.exists(os.path.join(path, 'processed')):
        os.mkdir(os.path.join(path, 'processed'))

    json_files = sorted([f for f in os.listdir(path) if f.endswith('.json')])
    df_list = []

    for json_file in json_files:
        file_path = os.path.join(path, json_file)
        with open(file_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df['source'] = json_file
        df_list.append(df)

        shutil.move(file_path, os.path.join(path, 'processed', json_file))

    df_processed = pd.concat(df_list, ignore_index=True)

    return df_processed