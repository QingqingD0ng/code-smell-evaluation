import pytz
from datetime import datetime

def _fromutc(dt, from_tz, to_tz):
    # Convert the input datetime to the timezone of the original datetime
    from_tz = pytz.timezone(from_tz)
    dt_with_tz = from_tz.localize(dt)
    
    # Convert the timezone datetime to the desired timezone
    to_tz = pytz.timezone(to_tz)
    dt_with_new_tz = dt_with_tz.astimezone(to_tz)
    
    return dt_with_new_tz

# Example usage:
if __name__ == "__main__":
    from_timestamp = datetime(2023, 4, 1, 12, 0)
    from_timezone = 'UTC'
    to_timezone = 'America/New_York'
    
    converted_dt = _fromutc(from_timestamp, from_timezone, to_timezone)
    print(converted_dt)