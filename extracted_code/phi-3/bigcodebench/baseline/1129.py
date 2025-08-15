import json
import requests
from datetime import datetime
import os

def task_func(json_data, unknown_key, save_dir=None):
    data = json.loads(json_data)
    url = data.get(unknown_key, None)
    if not url:
        raise ValueError(f"URL not found for key: {unknown_key}")

    response = requests.get(url)
    if response.status_code!= 200:
        raise ValueError(f"Failed to download file from {url}")

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
    filename = f"{unknown_key}_{timestamp}.txt"
    if save_dir:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, filename)
    else:
        filepath = os.path.join(os.getcwd(), filename)

    with open(filepath, 'wb') as file:
        file.write(response.content)

    return os.path.abspath(filepath)