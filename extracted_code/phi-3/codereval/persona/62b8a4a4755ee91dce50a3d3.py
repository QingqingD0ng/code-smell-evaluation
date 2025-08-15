import pytz
from datetime import datetime

def fromutc(self, dt, new_tz):
    # Check if the datetime object is timezone aware
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise ValueError("The provided datetime object is not timezone-aware.")

    # Convert the datetime object to UTC
    utc_dt = dt.astimezone(pytz.utc)

    # Check for ambiguous times ('fold' issue)
    if utc_dt.tzinfo.dst(utc_dt)!= datetime.timedelta(0):
        raise ValueError("The datetime object is ambiguous (fall-back or daylight saving time).")

    # Create a new timezone object
    new_tz = pytz.timezone(new_tz)

    # Convert the datetime object to the new timezone
    new_dt = utc_dt.astimezone(new_tz)

    return new_dt