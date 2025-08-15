import pandas as pd
import pytz
from datetime import datetime
from random import seed as set_seed

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
    },
    seed=42
):
    if not isinstance(utc_datetime, datetime):
        raise ValueError("utc_datetime must be a datetime object")
    if not isinstance(cities, list) or not all(isinstance(city, str) for city in cities):
        raise ValueError("cities must be a list of strings")
    if not isinstance(weather_conditions, list) or not all(isinstance(condition, str) for condition in weather_conditions):
        raise ValueError("weather_conditions must be a list of strings")
    if not isinstance(timezones, dict) or not all(isinstance(city, str) and isinstance(tz, str) for city, tz in timezones.items()):
        raise ValueError("timezones must be a dictionary with string keys and values")

    set_seed(seed)
    weather_report = []
    for city in cities:
        local_time = utc_datetime.astimezone(pytz.timezone(timezones[city]))
        condition = random.choice(weather_conditions)
        weather_report.append({'City': city, 'Local Time': local_time.strftime('%Y-%m-%d %H:%M:%S %Z'), 'Weather Condition': condition})

    return pd.DataFrame(weather_report)