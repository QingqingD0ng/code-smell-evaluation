from datetime import timedelta
import re

def parse_frequency(frequency_str):
    if frequency_str in (None, "always", ""):
        return None

    match = re.fullmatch(r'^(\d+)\s*(\w+)$', frequency_str)
    if not match:
        raise ValueError("Cannot parse frequency")

    value, unit = match.groups()
    try:
        value = int(value)
    except ValueError:
        raise ValueError("Frequency value must be an integer")

    time_units = {
      'second': timedelta(seconds=value),
      'minute': timedelta(minutes=value),
        'hour': timedelta(hours=value),
        'day': timedelta(days=value),
        'week': timedelta(weeks=value),
      'month': timedelta(days=value * 30),  # Months approximated to 30 days
        'year': timedelta(days=value * 365)  # Years approximated to 365 days
    }

    if unit not in time_units:
        raise ValueError("Unsupported time unit: '{}'".format(unit))

    return time_units[unit]