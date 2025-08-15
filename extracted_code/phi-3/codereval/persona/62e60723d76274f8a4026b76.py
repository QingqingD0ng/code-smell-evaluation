from datetime import datetime, timedelta, timezone

class Time:
    def __init__(self, naive_datetime):
        self.datetime = naive_datetime.replace(tzinfo=timezone.utc)

    @classmethod
    def from_ticks(cls, ticks, tz=None):
        if not (0 <= ticks < 86400000000000):
            raise ValueError("ticks is out of bounds")
        seconds = ticks / 1e9
        midnight = datetime.combine(datetime.now().date() - timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
        naive_datetime = midnight + timedelta(seconds=seconds)
        return cls(naive_datetime)

# Example usage:
# time_instance = Time.from_ticks(1234567890000000000)