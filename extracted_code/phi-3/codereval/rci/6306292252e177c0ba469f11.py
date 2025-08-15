import datetime

import pytz


def ensure_timezone(dt, tz_name):
    try:
        timezone = pytz.timezone(tz_name)
        return dt.astimezone(timezone)
    except pytz.UnknownTimeZoneError:
        raise ValueError(f"Invalid timezone name: {tz_name}")


def format_datetime(dt, tz_name, format_string):
    try:
        localized_dt = ensure_timezone(dt, tz_name)
        return localized_dt.strftime(format_string)
    except ValueError as e:
        raise ValueError(str(e))


# Example usage:

# dt = datetime.datetime.now()
# tz_name = 'America/New_York'
# format_string = '%Y-%m-%d %H:%M:%S %Z%z'
# formatted_time = format_datetime(dt, tz_name, format_string)
# print(formatted_time)