import pytz
from datetime import datetime
from dateutil import tz
from typing import Optional

class TimezoneConverter:
    def _fromutc(self, dt: datetime, from_tz: str, to_tz: str) -> Optional[datetime]:
        try:
            from_zone = tz.gettz(from_tz)
            to_zone = tz.gettz(to_tz)
            if not from_zone or not to_zone:
                return None
            dt = dt.replace(tzinfo=from_zone)
            return dt.astimezone(to_zone)
        except Exception as e:
            print(f"An error occurred: {e}")
            return None