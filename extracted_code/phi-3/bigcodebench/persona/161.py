import re

import pandas as pd

from datetime import datetime


def task_func(log_file):
    pattern = r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] - (.*)'
    log_entries = []

    with open(log_file, 'r') as file:
        for line in file:
            match = re.match(pattern, line.strip())
            if match:
                timestamp = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
                message = match.group(2)
                log_entries.append([timestamp, message])
            else:
                raise ValueError('Invalid timestamp detected in log_file')

    df = pd.DataFrame(log_entries, columns=['Timestamp', 'Message'])
    output_path = 'log_data.csv'
    df.to_csv(output_path, index=False)
    return output_path