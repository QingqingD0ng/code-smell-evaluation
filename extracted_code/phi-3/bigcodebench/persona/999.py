import urllib.request
import os
import csv
import collections

def task_func(url, column_name, csv_file_path):
    # Ensure the column name is in the CSV header
    if column_name not in csv.DictReader(open(csv_file_path)).fieldnames:
        os.remove(csv_file_path)
        raise ValueError(f"The provided column_name '{column_name}' does not exist in the CSV file.")
    
    # Download the CSV file
    with urllib.request.urlopen(url) as response, open(csv_file_path, 'wb') as out_file:
        data = response.read()
        out_file.write(data)
    
    # Count occurrences of each value in the specified column
    occurrences = collections.Counter()
    with open(csv_file_path, mode='r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            occurrences[row[column_name]] += 1
    
    os.remove(csv_file_path)  # Delete the file after processing
    return dict(occurrences)