from datetime import datetime
import pytz

class TimezoneConverter:
    def fromutc(self, dt, target_tz):
        if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
            raise ValueError("Input datetime must be timezone-aware.")
        localized_dt = dt.astimezone(pytz.timezone(target_tz))
        return localized_dt

converter = TimezoneConverter()
aware_dt = datetime.now(pytz.utc)
localized_dt = converter.fromutc(aware_dt, 'America/New_York')