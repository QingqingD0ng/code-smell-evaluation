import json
import requests
from datetime import datetime
import os

def download_file_with_timestamp(url_data, key, directory=None):
    data = json.loads(url_data)
    url = data.get(key, None)
    if not url:
        raise ValueError(f"URL not found for key: {key}")

    response = requests.get(url)
    if response.status_code!= 200:
        raise ValueError(f"Failed to download file from {url}")

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
    filename = f"{key}_{timestamp}.txt"
    if directory:
        if not os.path.isdir(directory):
            os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)
    else:
        filepath = os.path.join(os.getcwd(), filename)

    with open(filepath, 'wb') as file:
        file.write(response.content)

    return os.path.abspath(filepath)