import datetime
import pytz

def hydrate_time(nanoseconds, tz=None):
    seconds, nanoseconds = divmod(nanoseconds, 1_000_000_000)
    milliseconds, nanoseconds = divmod(nanoseconds, 1_000_000)
    microseconds, nanoseconds = divmod(nanoseconds, 1_000)
    
    dt = datetime.datetime(1, 1, 1, 0, 0, 0, microsecond=microseconds) + datetime.timedelta(
        seconds=seconds, milliseconds=milliseconds
    )
    
    if tz:
        tzinfo = pytz.timezone(tz)
        dt = dt.replace(tzinfo=pytz.utc).astimezone(tzinfo)
    
    return dt