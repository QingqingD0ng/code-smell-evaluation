from datetime import datetime
import pytz

class TimezoneConverter:
    def convert_timezone(self, dt, new_timezone):
        # Convert timezone-aware datetime object
        if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
            dt = pytz.utc.localize(dt)
        new_dt = dt.astimezone(pytz.timezone(new_timezone))

        # Handle ambiguous datetimes due to DST transitions
        if new_dt.fold == 1:
            new_dt = new_dt.replace(fold=0)

        return new_dt

# Example usage
# converter = TimezoneConverter()
# dt = datetime(2023, 3, 12, 1, 30)  # Example datetime without timezone
# new_dt = converter.convert_timezone(dt, 'US/Eastern')
# print(new_dt)