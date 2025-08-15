import pytz
from datetime import datetime

def from_utc(dt, new_tz):
    # Set the timezone for the given datetime object to UTC
    dt = dt.replace(tzinfo=pytz.utc)
    
    # Convert the datetime object to the new timezone
    new_dt = dt.astimezone(pytz.timezone(new_tz))
    
    return new_dt

# Example usage:
# from_utc(datetime(2023, 1, 1, 12, 0, 0), 'America/New_York')