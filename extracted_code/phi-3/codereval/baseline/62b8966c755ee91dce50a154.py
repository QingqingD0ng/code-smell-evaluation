import re
from datetime import datetime, timedelta, time, timezone
from dateutil import parser, tz

class ISO8601Parser:
    ISO8601_DATETIME_REGEX = re.compile(
        r'^(?P<year>\d{4})'
        r'(?:-(?P<month>\d{2}))?'
        r'(?:-(?P<day>\d{2}))?'
        r'(?:'
        r'T'
        r'(?P<hour>\d{2})'
        r':(?P<minute>\d{2})'
        r':(?P<second>\d{2})'
        r'(?:\.'
        r'(?P<subsecond>\d{1,6})'
        r')?'
        r')?'
        r'(?P<tz_sign>[+-])'
        r'(?P<tz_hour>\d{2})'
        r':?(?P<tz_minute>\d{2})?'
        r'$'
    )

    def isoparse(self, dt_str):
        match = self.ISO8601_DATETIME_REGEX.match(dt_str)
        if not match:
            raise ValueError("Invalid ISO-8601 datetime string")

        parts = match.groupdict()
        year, month, day = (int(parts[key]) for key in ('year','month', 'day'))
        hour, minute, second = (int(parts[key]) for key in ('hour','minute','second'))
        subsecond = float(parts['subsecond']) if parts['subsecond'] else 0

        if month is None:
            month, day = None, int(day)
        elif day is None:
            day, month = int(day), int(month)

        tz_sign, tz_hour, tz_minute = parts['tz_sign'], int(parts['tz_hour']), int(parts['tz_minute']) if parts['tz_minute'] else 0
        tz_offset = timedelta(hours=tz_hour, minutes=tz_