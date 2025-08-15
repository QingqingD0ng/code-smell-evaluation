import urllib.request
import os
import csv
from collections import Counter

def task_func(url, column_name, csv_file_path):
    try:
        with urllib.request.urlopen(url) as response, open(csv_file_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
    except Exception as e:
        os.remove(csv_file_path)
        raise RuntimeError(f"Failed to download CSV file: {e}")

    try:
        with open(csv_file_path, 'r') as file:
            reader = csv.DictReader(file)
            if column_name not in reader.fieldnames:
                os.remove(csv_file_path)
                raise ValueError(f"The provided column_name '{column_name}' does not exist in the CSV file.")

            column_values = [row[column_name] for row in reader]
            return dict(Counter(column_values))
    finally:
        os.remove(csv_file_path)

# Example usage:
# task_func('http://example.com/data.csv', 'category', 'downloaded_data.csv')