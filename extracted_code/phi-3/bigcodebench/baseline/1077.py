from datetime import datetime
import pytz
import numpy as np

def task_func(time_strings, timezone):
    if len(time_strings) < 2:
        return 0.0
    
    tz = pytz.timezone(timezone)
    converted_times = []
    
    for time_str in time_strings:
        dt = datetime.strptime(time_str, '%d/%m/%y %H:%M:%S.%f')
        converted_times.append(tz.localize(dt))
    
    diffs = np.diff([t.timestamp() for t in converted_times])
    
    return np.mean(diffs)