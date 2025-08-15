import re
import pandas as pd
from datetime import datetime

def task_func(log_file):
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
    
    output_path = 'log_data.csv'
    df.to_csv(output_path, index=False)
    
    return output_path