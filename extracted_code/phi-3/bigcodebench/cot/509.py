import pandas as pd
import csv
from difflib import SequenceMatcher

def task_func(file_path1, file_path2, delimiter=',', quotechar='"'):
    try:
        with open(file_path1, 'r') as file1, open(file_path2, 'r') as file2:
            reader1 = csv.reader(file1, delimiter=delimiter, quotechar=quotechar)
            reader2 = csv.reader(file2, delimiter=delimiter, quotechar=quotechar)
            
            lines1 = list(reader1)
            lines2 = list(reader2)
            
            if not lines1 or not lines2:
                raise ValueError("One or both files are empty.")
            
            differences = []
            for num, (line1, line2) in enumerate(zip(lines1, lines2), start=1):
                sm = SequenceMatcher(None, line1, line2)
                match = sm.get_matching_blocks()
                if not match:
                    if line1!= line2:
                        differences.append((num, '-', line1))
                    if line2!= line1:
                        differences.append((num, '+', line2))
                else:
                    differences.append((num,'', line1))
            
            return pd.DataFrame(differences, columns=['Line Number', 'Status', 'Content'])
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e}")
    except ValueError as e:
        raise ValueError(f"Value error: {e}")
    except Exception as e:
        raise Exception(f"An error occurred: {e}")