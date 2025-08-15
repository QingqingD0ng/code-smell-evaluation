import urllib.request
import os
import csv
import collections

def download_file(url, csv_file_path):
    if os.path.exists(csv_file_path):
        return
    try:
        with urllib.request.urlopen(url) as response, open(csv_file_path, 'wb') as out_file:
            out_file.write(response.read())
    except urllib.error.URLError as e:
        os.remove(csv_file_path)
        raise e
    except urllib.error.HTTPError as e:
        os.remove(csv_file_path)
        raise e
    except urllib.error.ContentTooShortError as e:
        os.remove(csv_file_path)
        raise e

def process_csv(csv_file_path, column_name):
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"The file '{csv_file_path}' does not exist.")
    
    try:
        with open(csv_file_path, mode='r') as infile:
            reader = csv.DictReader(infile)
            if column_name not in reader.fieldnames:
                raise ValueError(f"The provided column_name '{column_name}' does not exist in the CSV file '{csv_file_path}'.")
            counts = {}
            for row in reader:
                value = row[column_name]
                if value not in counts:
                    counts[value] = 1
                else:
                    counts[value] += 1
            return counts
    finally:
        os.remove(csv_file_path)

def task_func(url, column_name, csv_file_path):
    download_file(url, csv_file_path)
    return process_csv(csv_file_path, column_name)