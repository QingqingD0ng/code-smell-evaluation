import datetime
import pytz

class Time:
    def __init__(self, dt):
        self.dt = dt

    @classmethod
    def from_ticks(cls, ticks, tz=None):
        if not (0 <= ticks < 86400000000000):
            raise ValueError("ticks is out of bounds")

        seconds = ticks // 1_000_000_000
        nanoseconds = ticks % 1_000_000_000
        dt = datetime.datetime(1, 1, 1, 0, 0, 0) + datetime.timedelta(seconds=seconds)
        dt = dt.replace(microsecond=int(nanoseconds / 1000))

        if tz:
            dt = tz.localize(dt)
        return cls(dt)

# Example usage:
# time_instance = Time.from_ticks(5_250_000_000_000, pytz.timezone('UTC'))