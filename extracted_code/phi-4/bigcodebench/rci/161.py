import re
import pandas as pd
from datetime import datetime
import os

def extract_log_data(log_file, output_file='log_data.csv'):
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"The log file {log_file} does not exist.")
    
    log_pattern = r"^(?P<type>\w+): \[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] - (?P<message>.+)$"
    entries = []
    
    with open(log_file, 'r') as file:
        for line in file:
            match = re.match(log_pattern, line)
            if match:
                try:
                    timestamp = datetime.strptime(match.group('timestamp'), '%Y-%m-%d %H:%M:%S')
                    entries.append({
                        'Type': match.group('type'),
                        'Timestamp': timestamp,
                        'Message': match.group('message')
                    })
                except ValueError:
                    raise ValueError("Invalid timestamp format in log entry")
    
    if not entries:
        raise ValueError("No valid log entries found")
    
    df = pd.DataFrame(entries)
    df['Timestamp'] = df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df.to_csv(output_file, index=False)
    
    return output_file