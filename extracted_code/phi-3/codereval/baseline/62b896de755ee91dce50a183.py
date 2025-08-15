import datetime
from dateutil import parser
from dateutil.tz import gettz

class CustomParser:
    def parse(self, timestr, default=None, ignoretz=False, tzinfos=None, **kwargs):
        return parser.parse(timestr, default=default, ignoretz=ignoretz, tzinfos=tzinfos, **kwargs)