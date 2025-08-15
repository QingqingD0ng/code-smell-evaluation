import re
import pandas as pd
from datetime import datetime

def task_func(log_file):
    log_pattern = r'^(?P<type>\w+):\s+\[(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\]\s+-\s+(?P<message>.+)$'
    data = []

    with open(log_file, 'r') as file:
        for line in file:
            match = re.match(log_pattern, line.strip())
            if match:
                try:
                    dt = datetime.strptime(match.group('timestamp'), '%Y-%m-%d %H:%M:%S')
                    data.append({
                        'Type': match.group('type'),
                        'Timestamp': dt,
                        'Message': match.group('message')
                    })
                except ValueError:
                    raise ValueError("Invalid timestamp format encountered.")

    if not data:
        raise ValueError("No valid log entries found.")

    df = pd.DataFrame(data)
    output_path = 'log_data.csv'
    df.to_csv(output_path, index=False, columns=['Type', 'Timestamp', 'Message'])
    return output_path