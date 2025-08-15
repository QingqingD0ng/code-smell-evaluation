import datetime

def parse_frequency(frequency):
    if frequency in (None, "always"):
        return None

    number, _, timeunit = frequency.partition(' ')
    timeunit = timeunit.lower()

    time_units_to_seconds = {
       'second': 1,
       'minute': 60,
        'hour': 3600,
        'day': 86400,
        'week': 604800,
       'month': 2629800,  # Approximate average month length in seconds
        'year': 31557600,  # Approximate average year length in seconds
    }

    if timeunit not in time_units_to_seconds:
        raise ValueError(f"Unknown time unit: {timeunit}")

    seconds = int(number) * time_units_to_seconds[timeunit]
    return datetime.timedelta(seconds=seconds)