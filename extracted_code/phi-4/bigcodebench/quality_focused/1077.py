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

    time_differences = [
        abs((converted_times[i] - converted_times[i - 1]).total_seconds()) 
        for i in range(1, len(converted_times))
    ]

    return np.mean(time_differences) if time_differences else 0.0