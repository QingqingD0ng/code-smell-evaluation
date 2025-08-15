import pandas as pd
import pytz
from datetime import datetime
from random import randint, seed as set_seed

def task_func(
    utc_datetime,
    cities=None,
    weather_conditions=None,
    timezones=None,
    seed=42
):
    if cities is None:
        cities = ['New York', 'London', 'Beijing', 'Tokyo', 'Sydney']
    if weather_conditions is None:
        weather_conditions = ['Sunny', 'Cloudy', 'Rainy', 'Snowy', 'Stormy']
    if timezones is None:
        timezones = {
            'New York': 'America/New_York',
            'London': 'Europe/London',
            'Beijing': 'Asia/Shanghai',
            'Tokyo': 'Asia/Tokyo',
            'Sydney': 'Australia/Sydney'
        }

    set_seed(seed)

    if not isinstance(utc_datetime, datetime) or utc_datetime.tzinfo!= pytz.UTC:
        raise ValueError("utc_datetime must be a timezone-aware datetime object set to UTC.")

    data = []
    for city in cities:
        if city not in timezones:
            continue
        local_tz = pytz.timezone(timezones[city])
        local_time = utc_datetime.astimezone(local_tz)
        weather_condition = weather_conditions[randint(0, len(weather_conditions) - 1)]
        data.append((city, local_time.strftime('%Y-%m-%d %H:%M:%S %Z'), weather_condition))

    return pd.DataFrame(data, columns=['City', 'Local Time', 'Weather Condition'])