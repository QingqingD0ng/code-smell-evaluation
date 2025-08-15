from datetime import datetime
import pytz
import numpy as np

def task_func(time_strings, timezone):
    if len(time_strings) < 2:
        return 0.0

    tz = pytz.timezone(timezone)
    times = [datetime.strptime(ts, '%d/%m/%y %H:%M:%S.%f').replace(tzinfo=pytz.utc).astimezone(tz) for ts in time_strings]
    differences = np.diff([t.timestamp() for t in times])
    return np.mean(np.abs(differences))