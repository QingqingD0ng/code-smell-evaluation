import datetime
import pytz
import numpy as np

def task_func(time_strings, timezone):
    if len(time_strings) < 2:
        return 0.0

    tz = pytz.timezone(timezone)
    times = [tz.localize(datetime.datetime.strptime(ts, '%d/%m/%y %H:%M:%S.%f')) for ts in time_strings]
    time_diffs = np.diff([t.timestamp() for t in times]).astype(float)
    return np.mean(time_diffs)