from datetime import datetime
import pytz
import numpy as np

def task_func(time_strings, timezone):
    if len(time_strings) < 2:
        return 0.0

    tz = pytz.timezone(timezone)
    converted_times = [
        datetime.strptime(ts, '%d/%m/%y %H:%M:%S.%f').replace(tzinfo=pytz.utc).astimezone(tz)
        for ts in time_strings
    ]

    time_diffs = [
        (converted_times[i+1] - converted_times[i]).total_seconds()
        for i in range(len(converted_times) - 1)
    ]

    return np.mean(time_diffs)