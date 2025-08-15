from datetime import datetime
import pytz
import numpy as np

DATE_FORMAT = '%d/%m/%y %H:%M:%S.%f'

def task_func(time_strings, timezone):
    if len(time_strings) < 2:
        return 0.0

    try:
        tz = pytz.timezone(timezone)
    except pytz.UnknownTimeZoneError:
        raise ValueError("Invalid timezone provided")

    converted_times = []
    for ts in time_strings:
        try:
            dt = datetime.strptime(ts, DATE_FORMAT)
            dt = dt.replace(tzinfo=pytz.utc).astimezone(tz)
            converted_times.append(dt)
        except ValueError:
            raise ValueError(f"Invalid date format: {ts}")

    differences = np.diff([t.timestamp() for t in converted_times])
    mean_diff = np.mean(np.abs(differences))
    return mean_diff