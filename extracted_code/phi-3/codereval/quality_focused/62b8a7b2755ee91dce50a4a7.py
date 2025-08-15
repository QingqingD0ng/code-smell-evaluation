from datetime import datetime
from dateutil import tz
from dateutil import parser

def default_tzinfo(dt, tzinfo):
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tzinfo)
    return dt