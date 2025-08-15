from pytz import timezone
from datetime import datetime

def ensure_timezone(dt, tz_name):
    tz = timezone(tz_name)
    return dt.astimezone(tz)

def format_dt(dt):
    return ensure_timezone(dt, 'UTC').strftime('%Y-%m-%d %H:%M:%S')