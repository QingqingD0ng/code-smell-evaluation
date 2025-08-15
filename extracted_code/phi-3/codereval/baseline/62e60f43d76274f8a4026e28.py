import datetime
import pytz

def hydrate_time(nanoseconds, tz=None):
    # Convert nanoseconds to seconds
    seconds = nanoseconds / 1e9

    # Get current time
    now = datetime.datetime.now(pytz.utc)

    # Calculate the time the nanoseconds represent
    hydrated_time = now + datetime.timedelta(seconds=seconds)

    # Convert to the specified timezone or default to UTC if not provided
    if tz:
        tz_info = pytz.timezone(tz)
        hydrated_time = hydrated_time.astimezone(tz_info)
    else:
        hydrated_time = hydrated_time.astimezone(pytz.utc)

    # Format the time as a string in 'YYYY-MM-DD HH:MM:SS' format
    return hydrated_time.strftime('%Y-%m-%d %H:%M:%S')