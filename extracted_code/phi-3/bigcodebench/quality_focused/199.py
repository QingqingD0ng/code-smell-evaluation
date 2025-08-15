import pandas as pd
import pytz
from datetime import datetime, timedelta
from random import seed, choices

seed(42)

def task_func(
    utc_datetime,
    cities=['New York', 'London', 'Beijing', 'Tokyo', 'Sydney'],
    weather_conditions=['Sunny', 'Cloudy', 'Rainy', 'Snowy', 'Stormy'],
    timezones={
        'New York': 'America/New_York',
        'London': 'Europe/London',
        'Beijing': 'Asia/Shanghai',
        'Tokyo': 'Asia/Tokyo',
        'Sydney': 'Australia/Sydney'
    }
):
    if not isinstance(utc_datetime, datetime):
        raise ValueError("utc_datetime must be a datetime object.")

    city_local_time_weather = []
    for city in cities:
        local_timezone = pytz.timezone(timezones.get(city))
        local_datetime = utc_datetime + timedelta(hours=int(local_timezone.utcoffset(utc_datetime).total_seconds() / 3600))
        local_time = local_datetime.astimezone(local_timezone).strftime('%Y-%m-%d %H:%M:%S %Z')
        weather_condition = choices(weather_conditions)[0]
        city_local_time_weather.append((city, local_time, weather_condition))

    return pd.DataFrame(city_local_time_weather, columns=['City', 'Local Time', 'Weather Condition'])