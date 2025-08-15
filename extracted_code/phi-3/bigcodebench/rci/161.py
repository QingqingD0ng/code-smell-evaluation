import re
import pandas as pd
from datetime import datetime

def validate_timestamp(timestamp):
    try:
        datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        raise ValueError('Invalid timestamp format')

def parse_line(line, patterns):
    for pattern in patterns:
        match = re.match(pattern, line)
        if match:
            return {
                'Type': match.group(1),
                'Timestamp': validate_timestamp(match.group(2)),
                'Message': match.group(3)
            }
    raise ValueError('Invalid log entry format')

def task_func(log_file, patterns=None):
    if patterns is None:
        patterns = [r'(\w+): \[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] - (.*)']

    log_entries = []

    with open(log_file, 'r') as file:
        for line in file:
            entry = parse_line(line, patterns)
            log_entries.append(entry)

    if not log_entries:
        raise ValueError('No valid log entries found')

    df = pd.DataFrame(log_entries)

    csv_path = log_file.replace('.log', '_log_data.csv')
    df.to_csv(csv_path, index=False)

    return csv_path