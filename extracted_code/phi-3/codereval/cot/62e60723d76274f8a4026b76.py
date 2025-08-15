from datetime import datetime, timedelta, timezone

class Time:
    @classmethod
    def from_ticks(cls, ticks, tz=None):
        if not (0 <= ticks < 86400000000000):
            raise ValueError("ticks is out of bounds")

        total_seconds = ticks / 1_000_000_000
        midnight = datetime.combine(datetime.today(), datetime.min.time())
        time_since_midnight = midnight + timedelta(seconds=total_seconds)

        if tz is not None:
            time_since_midnight = time_since_midnight.replace(tzinfo=tz)

        return cls(time_since_midnight)

    def __init__(self, datetime_obj):
        self.datetime_obj = datetime_obj

# Example usage:
# tz = timezone(timedelta(hours=1))  # Example timezone
# time_obj = Time.from_ticks(1000000000, tz=tz)