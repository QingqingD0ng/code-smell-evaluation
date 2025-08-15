import datetime
import pytz

class TimezoneConverter:
    def _fromutc(self, dt, from_tz, to_tz):
        from_tz = pytz.timezone(from_tz)
        to_tz = pytz.timezone(to_tz)
        dt = from_tz.localize(dt)
        return dt.astimezone(to_tz)

converter = TimezoneConverter()
dt_utc = datetime.datetime(2023, 1, 1, 12, 0, 0, tzinfo=pytz.utc)
from_timezone = 'America/New_York'
to_timezone = 'Europe/London'
converted_dt = converter._fromutc(dt_utc, from_timezone, to_timezone)
print(converted_dt)