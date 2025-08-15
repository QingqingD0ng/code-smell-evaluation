import arrow

def hydrate_time(nanoseconds, tz=None):
    seconds = nanoseconds / 1e9
    now = arrow.utcnow()
    hydrated_time = now.shift(seconds=seconds)
    if tz:
        hydrated_time = hydrated_time.to(arrow.parser.to_datetime_string(tz))
    else:
        hydrated_time = hydrated_time.to('UTC')
    return hydrated_time.format('YYYY-MM-DDTHH:mm:ssZ')