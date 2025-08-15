import re
import pandas as pd
from datetime import datetime

def task_func(log_file):
    log_pattern = r'^(?P<type>\w+):\s+\[(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})\]\s+-\s+(?P<message>.+)$'
    log_entries = []

    with open(log_file, 'r') as file:
        for line in file:
            match = re.match(log_pattern, line)
            if match:
                log_data = match.groupdict()
                try:
                    log_data['timestamp'] = datetime.strptime(log_data['timestamp'], '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    raise ValueError("Invalid timestamp format found.")
                log_entries.append(log_data)

    if not log_entries:
        raise ValueError("No valid log entries found.")

    df = pd.DataFrame(log_entries)
    csv_file = 'log_data.csv'
    df.to_csv(csv_file, index=False, columns=['type', 'timestamp','message'])

    return csv_file