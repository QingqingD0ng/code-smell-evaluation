import re

import pandas as pd

from datetime import datetime


def task_func(log_file):
    pattern = r'(\w+): \[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] - (.*)'
    log_data = []

    with open(log_file, 'r') as file:
        for line in file:
            match = re.match(pattern, line)
            if match:
                message_type, timestamp, message = match.groups()
                try:
                    datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                    log_data.append((message_type, timestamp, message))
                except ValueError:
                    raise ValueError("Invalid timestamp format in log entry.")
            else:
                raise ValueError("Invalid log entry format.")

    df = pd.DataFrame(log_data, columns=['Type', 'Timestamp', 'Message'])
    output_file = 'log_data.csv'
    df.to_csv(output_file, index=False)
    return output_file