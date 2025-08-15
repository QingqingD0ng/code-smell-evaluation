import datetime
import pytz

class Time:
    MAX_TICKS = 86400000000000

    def __init__(self, dt):
        self.dt = dt

    @classmethod
    def from_ticks(cls, ticks: int, tz: str = None):
        if not (0 <= ticks < cls.MAX_TICKS):
            raise ValueError(f"ticks is out of bounds: {ticks} (expected 0 <= ticks < {cls.MAX_TICKS})")

        seconds = ticks // 1_000_000_000
        nanoseconds = ticks % 1_000_000_000
        dt = datetime.datetime(1, 1, 1, 0, 0, 0) + datetime.timedelta(seconds=seconds)
        dt = dt.replace(microsecond=int(nanoseconds / 1000))

        if tz:
            try:
                tz = pytz.timezone(tz)
            except pytz.UnknownTimeZoneError:
                raise ValueError(f"Invalid timezone string: {tz}")
            dt = tz.localize(dt)
        return cls(dt)

# Example usage:
# time_instance = Time.from_ticks(5_250_000_000_000, 'UTC')