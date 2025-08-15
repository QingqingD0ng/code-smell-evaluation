import pandas as pd
import json
import os
import shutil

def task_func(path):
    # Check if the directory exists, if not create it
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Get a list of JSON files sorted alphabetically
    json_files = sorted([f for f in os.listdir(path) if f.endswith('.json')])
    
    # Initialize an empty DataFrame
    df = pd.DataFrame()
    
    # Process each JSON file
    for json_file in json_files:
        file_path = os.path.join(path, json_file)
        with open(file_path, 'r') as f:
            data = json.load(f)
            # Convert JSON data to DataFrame, add "Source" column
            data_df = pd.DataFrame(data)
            data_df['source'] = json_file
            # Append to the main DataFrame
            df = df.append(data_df, ignore_index=True)
    
    # Move processed files to "processed" subdirectory
    processed_dir = os.path.join(path, 'processed')
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    for json_file in json_files:
        shutil.move(os.path.join(path, json_file), os.path.join(processed_dir, json_file))
    
    return df