import pandas as pd
import csv
from difflib import ndiff

def task_func(file_path1, file_path2, delimiter=',', quotechar='"'):
    # Read the CSV files
    try:
        with open(file_path1, newline='', encoding='utf-8') as file1, open(file_path2, newline='', encoding='utf-8') as file2:
            reader1 = csv.reader(file1, delimiter=delimiter, quotechar=quotechar)
            reader2 = csv.reader(file2, delimiter=delimiter, quotechar=quotechar)
            
            # Compare the CSV files line by line
            differences = []
            line_number = 1
            for row1, row2 in zip(reader1, reader2):
                if row1!= row2:
                    diff = ndiff(row1, row2)
                    diff_str = ''.join(diff).split('\n')
                    status = []
                    content = []
                    for d in diff_str:
                        if d.startswith('- ') or d.startswith('+ '):
                            status.append('-' if d.startswith('- ') else '+')
                            content.append(d[2:])
                        elif d.startswith('  '):
                            status.append(' ')
                            content.append(d[2:])
                    differences.append({
                        'Line Number': line_number,
                        'Status': status,
                        'Content': content
                    })
                line_number += 1

            # Convert the differences to a DataFrame
            return pd.DataFrame(differences)

    except FileNotFoundError as e:
        raise FileNotFoundError("One of the files cannot be found.") from e
    except ValueError as e:
        raise ValueError("One of the files is empty.") from e
    except Exception as e:
        raise Exception("An IO related error occurred.") from e