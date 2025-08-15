import json
import requests
import os
from datetime import datetime

def task_func(json_data, unknown_key, save_dir=None):
    data = json.loads(json_data)
    url = data.get(unknown_key)
    
    if not url:
        raise ValueError(f"Key '{unknown_key}' not found in JSON data.")
    
    response = requests.get(url)
    response.raise_for_status()
    
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
    filename = f"{unknown_key}_{timestamp}.txt"
    
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_path = os.path.join(save_dir, filename)
    else:
        file_path = filename
    
    with open(file_path, 'wb') as f:
        f.write(response.content)
    
    return os.path.abspath(file_path)