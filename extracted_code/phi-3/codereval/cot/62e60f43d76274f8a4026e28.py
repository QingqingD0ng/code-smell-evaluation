from datetime import datetime


def hydrate_time(nanoseconds, tz=None):
    seconds = nanoseconds / 1_000_000_000
    time_obj = datetime.fromtimestamp(seconds, tz=tz)
    return time_obj.strftime('%Y-%m-%d %H:%M:%S')