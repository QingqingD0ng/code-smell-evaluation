import csv
import os
from collections import Counter

CSV_FILE_PATH ='match_data.csv'

def create_test_csv(filename, content):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(content)

def task_func(goals, penalties, csv_file_path=CSV_FILE_PATH):
    count = Counter()
    if os.path.exists(csv_file_path):
        with open(csv_file_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                count['goals'] += int(row['goals'])
                count['penalties'] += int(row['penalties'])
                
    count['goals'] += sum(goals.values())
    count['penalties'] += sum(penalties.values())
    
    return count