import urllib.request
import os
import csv
import collections

def task_func(url, column_name, csv_file_path):
    try:
        # Download CSV file
        with urllib.request.urlopen(url) as response, open(csv_file_path, 'wb') as out_file:
            out_file.write(response.read())
    except Exception as e:
        os.remove(csv_file_path)
        raise e

    try:
        # Read CSV file
        with open(csv_file_path, mode='r') as infile:
            reader = csv.DictReader(infile)
            if column_name not in reader.fieldnames:
                raise ValueError(f"The provided column_name '{column_name}' does not exist in the CSV file.")
            counts = collections.Counter(row[column_name] for row in reader)
    finally:
        # Delete downloaded CSV file
        os.remove(csv_file_path)

    return dict(counts)