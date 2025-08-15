from datetime import timedelta
import re

def parse_frequency(frequency):
    if frequency in (None, "always"):
        return None
    match = re.match(r'^(\d+)\s*(\w+)$', frequency)
    if not match:
        raise ValueError("Cannot parse frequency")
    value, unit = match.groups()
    value = int(value)
    conversions = {
       'second': timedelta(seconds=value),
       'minute': timedelta(minutes=value),
        'hour': timedelta(hours=value),
        'day': timedelta(days=value),
        'week': timedelta(weeks=value),
       'month': timedelta(days=value*30),  # Approximation
        'year': timedelta(days=value*365),  # Approximation
    }
    if unit not in conversions:
        raise ValueError("Unsupported time unit")
    return conversions[unit]