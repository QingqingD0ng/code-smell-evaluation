import datetime
from dateutil import parser as dateutil_parser
from dateutil import tz

class ISO8601Parser:
    def isoparse(self, dt_str):
        dt = dateutil_parser.isoparse(dt_str)
        return dt