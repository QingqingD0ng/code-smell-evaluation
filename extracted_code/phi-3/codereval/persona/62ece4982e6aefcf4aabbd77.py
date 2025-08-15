import re
from datetime import timedelta

def parse_frequency(frequency):
    if frequency is None or frequency.lower() == "always":
        return None
    match = re.match(r'^(\d+)\s+(\w+)$', frequency)
    if not match:
        raise ValueError(f"Cannot parse frequency: {frequency}")
    number, unit = match.groups()
    number = int(number)
    unit_to_timedelta = {
       'second': timedelta(seconds=number),
       'minute': timedelta(minutes=number),
        'hour': timedelta(hours=number),
        'day': timedelta(days=number),
        'week': timedelta(weeks=number),
       'month': timedelta(days=number*30),  # Approximation
        'year': timedelta(days=number*365),  # Approximation
    }
    if unit not in unit_to_timedelta:
        raise ValueError(f"Invalid time unit: {unit}")
    return unit_to_timedelta[unit]