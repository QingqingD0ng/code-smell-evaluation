import pytz
from datetime import datetime
from dateutil import tz

class TimezoneConverter:
    def _fromutc(self, dt, from_tz, to_tz):
        from_zone = pytz.timezone(from_tz)
        to_zone = pytz.timezone(to_tz)
        dt = from_zone.localize(dt)
        return dt.astimezone(to_zone)