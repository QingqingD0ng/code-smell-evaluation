from datetime import datetime
import pytz

def ensure_timezone(dt, tz_name):
    timezone = pytz.timezone(tz_name)
    return dt.astimezone(timezone)

def format_dt(dt, tz_name='UTC'):
    return ensure_timezone(dt, tz_name).strftime('%Y-%m-%d %H:%M:%S %Z')