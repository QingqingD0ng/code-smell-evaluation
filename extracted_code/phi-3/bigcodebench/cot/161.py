import re

import pandas as pd

from datetime import datetime


def task_func(log_file):

    log_pattern = re.compile(r'(\w+): \[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] - (.+)')

    log_entries = []


    with open(log_file, 'r') as file:

        for line in file:

            match = log_pattern.search(line)

            if match:

                log_type, timestamp_str, message = match.groups()

                try:

                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')

                    log_entries.append({

                        'type': log_type,

                        'timestamp': timestamp,

                      'message': message.strip()

                    })

                except ValueError:

                    raise ValueError("Invalid timestamp format encountered.")

            else:

                raise ValueError("Invalid log entry format.")


    log_df = pd.DataFrame(log_entries)

    csv_output_path = log_file.replace('.log', '_log_data.csv')

    log_df.to_csv(csv_output_path, index=False)

    return csv_output_path