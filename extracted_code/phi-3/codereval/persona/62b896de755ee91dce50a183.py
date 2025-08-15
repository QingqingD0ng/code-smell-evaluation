from dateutil import parser
from datetime import datetime, tzinfo, timezone

def parse(self, timestr, default=None, ignoretz=False, tzinfos=None, **kwargs):
    kwargs['fuzzy_with_tokens'] = kwargs.get('fuzzy_with_tokens', False)
    if ignoretz:
        kwargs['ignoretz'] = True
    if tzinfos:
        kwargs['tzinfos'] = tzinfos
    dt = parser.parse(timestr, **kwargs)
    if default is not None and isinstance(default, datetime):
        dt = dt.replace(default)
    if not ignoretz:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt