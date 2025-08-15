import urllib.request
import os
import csv
from collections import Counter

def task_func(url, column_name, csv_file_path):
    if not os.path.exists(csv_file_path):
        try:
            with urllib.request.urlopen(url) as response, open(csv_file_path, 'wb') as out_file:
                data = response.read()
                out_file.write(data)
        except Exception as e:
            os.remove(csv_file_path)
            raise e

    try:
        with open(csv_file_path, mode='r', newline='') as in_file:
            reader = csv.DictReader(in_file)
            if column_name not in reader.fieldnames:
                os.remove(csv_file_path)
                raise ValueError(f"The provided column_name '{column_name}' does not exist in the CSV file.")

            counts = Counter(row[column_name] for row in reader)

        os.remove(csv_file_path)
        return dict(counts)
    except Exception as e:
        raise e