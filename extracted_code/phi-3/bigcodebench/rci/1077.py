from datetime import datetime
import pytz
import numpy as np

def task_func(time_strings, timezone):
    """
    Calculates the average time difference in seconds between each consecutive pair of timestamps
    in a given list, after converting them to a specified timezone.
    """
    if len(time_strings) < 2:
        return 0.0
    
    tz = pytz.timezone(timezone)
    timestamp_diffs = []
    
    for time_str in time_strings:
        dt = datetime.strptime(time_str, '%d/%m/%y %H:%M:%S.%f')
        localized_dt = tz.localize(dt)
        timestamp_diffs.append(localized_dt.timestamp())
    
    time_diffs = np.diff(timestamp_diffs)
    
    return np.nanmean(time_diffs) if time_diffs.size > 0 else 0.0