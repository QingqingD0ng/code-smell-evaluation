from datetime import timedelta

def parse_frequency(frequency):
    if frequency is None or frequency == "always":
        return None
    number, unit = frequency.split()
    number = int(number)
    if unit in ["second", "minute", "hour", "day", "week", "month", "year"]:
        return timedelta(**{unit: number})
    raise ValueError(f"Unknown time unit: {unit}")