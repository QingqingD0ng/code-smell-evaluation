import datetime
import pytz

def hydrate_time(nanoseconds, tz=None):
    # Calculate seconds, microseconds and nanoseconds
    seconds, microseconds, nanoseconds = divmod(nanoseconds, 1_000_000_000)
    # Create a datetime object
    dt = datetime.datetime(1, 1, 1, tzinfo=pytz.UTC) + datetime.timedelta(seconds=seconds)
    dt = dt.replace(microsecond=microseconds)
    # Append nanoseconds if any
    dt = dt.replace(microsecond=dt.microsecond % 1000_000 + (nanoseconds // 1) % 1000)
    
    # Apply timezone if specified
    if tz:
        tz_obj = pytz.timezone(tz)
        dt = tz_obj.localize(dt)
    
    return dt