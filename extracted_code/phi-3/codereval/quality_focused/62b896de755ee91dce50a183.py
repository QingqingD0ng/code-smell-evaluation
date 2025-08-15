from datetime import datetime
from dateutil.parser import parse as dateutil_parse
from dateutil.tz import tz

class Parser:
    def parse(self, timestr, default=None, ignoretz=False, tzinfos=None, **kwargs):
        try:
            dt = dateutil_parse(timestr, default=default, ignoretz=ignoretz, tzinfos=tzinfos, **kwargs)
            if ignoretz and dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)
            return dt
        except (ValueError, OverflowError) as e:
            raise e
        except TypeError:
            raise TypeError("Input must be a string or character stream")