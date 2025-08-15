import urllib.request
import os
import csv
import collections

def task_func(url, column_name, csv_file_path):
    try:
        # Download the CSV file
        urllib.request.urlretrieve(url, csv_file_path)
        
        # Open the CSV file and read data
        with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            # Check if the column exists
            if column_name not in reader.fieldnames:
                os.remove(csv_file_path)
                raise ValueError(f"The provided column_name '{column_name}' does not exist in the CSV file.")
            
            # Count occurrences of each value in the specified column
            counter = collections.Counter()
            for row in reader:
                counter[row[column_name]] += 1
        
        # Delete the downloaded file
        os.remove(csv_file_path)
        
        return dict(counter)
    
    except urllib.error.URLError as e:
        if os.path.exists(csv_file_path):
            os.remove(csv_file_path)
        raise RuntimeError(f"Failed to download the file: {e.reason}")