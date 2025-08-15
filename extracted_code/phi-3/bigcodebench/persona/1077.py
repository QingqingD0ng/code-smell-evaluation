from datetime import datetime

import pytz

import numpy as np


def task_func(time_strings, timezone):
    if len(time_strings) < 2:
        return 0.0
    tzs = pytz.timezone(timezone)
    times = [tzs.localize(datetime.strptime(t, '%d/%m/%y %H:%M:%S.%f')) for t in time_strings]
    differences = [int((times[i + 1] - times[i]).total_seconds()) for i in range(len(times) - 1)]
    return np.mean(differences)