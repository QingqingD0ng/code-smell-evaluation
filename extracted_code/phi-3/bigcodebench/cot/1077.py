from datetime import datetime
import pytz
import numpy as np

def task_func(time_strings, timezone):
    tz = pytz.timezone(timezone)
    times = [tz.localize(datetime.strptime(ts, '%d/%m/%y %H:%M:%S.%f')) for ts in time_strings]
    if len(times) < 2:
        return 0.0
    diffs = np.diff([t.timestamp() for t in times])
    return np.mean(diffs)