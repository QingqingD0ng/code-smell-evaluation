import re
import pandas as pd
from datetime import datetime

def task_func(log_file):
    pattern = r'(\w+): \[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] - (.*)'
    with open(log_file, 'r') as file:
        lines = file.readlines()

    log_entries = []
    for line in lines:
        match = re.match(pattern, line)
        if match:
            log_entries.append({
                'Type': match.group(1),
                'Timestamp': datetime.strptime(match.group(2), '%Y-%m-%d %H:%M:%S'),
                'Message': match.group(3)
            })
        else:
            raise ValueError('Invalid log entry format')

    if not log_entries:
        raise ValueError('No valid log entries found')

    df = pd.DataFrame(log_entries)
    csv_path = log_file.replace('.log', '_log_data.csv')
    df.to_csv(csv_path, index=False)

    return csv_path