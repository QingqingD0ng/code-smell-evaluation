class Time:
    def __init__(self, nanoseconds, tz=None):
        self._nanoseconds = nanoseconds
        self._tz = tz

    @classmethod
    def from_ticks(cls, ticks, tz=None):
        if not (0 <= ticks < 86400000000000):
            raise ValueError("ticks out of bounds")
        # Convert ticks to datetime
        seconds = ticks / 1_000_000_000
        return cls(seconds, tz)

    # Other methods (e.g., to_string, to_timestamp) would be here