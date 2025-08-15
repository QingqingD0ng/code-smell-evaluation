import urllib.request
import os
import csv
import collections

def task_func(url, column_name, csv_file_path):
    try:
        # Download the CSV file from the given URL
        urllib.request.urlretrieve(url, csv_file_path)

        # Initialize a counter for the column values
        counter = collections.Counter()

        # Open the CSV file and read its contents
        with open(csv_file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Check if the specified column exists in the CSV file
            if column_name not in reader.fieldnames:
                os.remove(csv_file_path)
                raise ValueError(f"The provided column_name '{column_name}' does not exist in the CSV file.")
            
            # Count occurrences of each value in the specified column
            for row in reader:
                counter[row[column_name]] += 1

        # Delete the downloaded CSV file after processing
        os.remove(csv_file_path)

        # Return the dictionary of occurrences
        return dict(counter)

    except urllib.error.URLError as e:
        if os.path.exists(csv_file_path):
            os.remove(csv_file_path)
        print(f"Error downloading the file: {e}")
        return {}