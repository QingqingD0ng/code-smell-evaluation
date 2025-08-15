import datetime
from dateutil import parser, tz

def isoparse(dt_str):
    try:
        return parser.isoparse(dt_str)
    except ValueError as e:
        raise ValueError(f"Invalid ISO-8601 datetime string: {dt_str}") from e