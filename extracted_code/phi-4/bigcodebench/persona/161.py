import re
import pandas as pd
from datetime import datetime

def task_func(log_file):
    log_pattern = re.compile(r'^(?P<type>\w+):\s+\[(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\]\s+-\s+(?P<message>.+)$')
    data = []

    with open(log_file, 'r') as file:
        for line in file:
            match = log_pattern.match(line)
            if match:
                log_info = match.groupdict()
                try:
                    datetime.strptime(log_info['timestamp'], '%Y-%m-%d %H:%M:%S')
                    data.append(log_info)
                except ValueError:
                    raise ValueError("Invalid timestamp format found.")

    if not data:
        raise ValueError("No valid log entries found.")

    df = pd.DataFrame(data)
    output_path = 'log_data.csv'
    df.to_csv(output_path, index=False)

    return output_path