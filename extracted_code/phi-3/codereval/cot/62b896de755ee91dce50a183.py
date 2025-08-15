import dateutil.parser as dparser
from dateutil.tz import gettz, tzfile

class DateParser:
    def parse(self, timestr, default=None, ignoretz=False, tzinfos=None, **kwargs):
        if not isinstance(timestr, str):
            raise TypeError("Input must be a string")

        if default is not None and not isinstance(default, datetime.datetime):
            raise TypeError("Default must be a datetime object or None")

        if tzinfos is not None:
            if not isinstance(tzinfos, dict) and not callable(tzinfos):
                raise TypeError("tzinfos must be a dictionary or a callable")

        try:
            dt = dparser.parse(timestr, default=default, ignoretz=ignoretz, tzinfos=tzinfos, **kwargs)
            return dt
        except (ValueError, OverflowError) as e:
            raise ParserError("Invalid or unknown string format") from e
        except AttributeError as e:
            raise TypeError("Invalid type for default or tzinfos") from e

# Example usage:
# parser = DateParser()
# dt_obj = parser.parse("2012-01-19 17:21:00", tzinfos={"BRST": -7200, "CST": gettz("America/Chicago")})