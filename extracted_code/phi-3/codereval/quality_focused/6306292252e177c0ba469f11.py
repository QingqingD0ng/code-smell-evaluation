from datetime import datetime
import pytz

def format_dt(dt):
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise ValueError("Provided datetime object does not have timezone information")
    return dt.astimezone(pytz.utc).replace(tzinfo=None)