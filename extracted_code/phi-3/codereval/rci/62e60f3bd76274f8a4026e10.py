from datetime import timedelta

class TimeStructure:
    SECONDS_PER_MINUTE = 60
    MINUTES_PER_HOUR = 60
    HOURS_PER_DAY = 24

    def __init__(self, days, hours, minutes, seconds):
        self.days = days
        self.hours = hours
        self.minutes = minutes
        self.seconds = seconds

def dehydrate_timedelta(value):
    total_seconds = int(value.total_seconds())
    days, remainder = divmod(total_seconds, TimeStructure.HOURS_PER_DAY * TimeStructure.MINUTES_PER_HOUR * TimeStructure.SECONDS_PER_MINUTE)
    hours, remainder = divmod(remainder, TimeStructure.MINUTES_PER_HOUR * TimeStructure.SECONDS_PER_MINUTE)
    minutes, seconds = divmod(remainder, TimeStructure.SECONDS_PER_MINUTE)
    return TimeStructure(days, hours, minutes, seconds)

# Demonstration of the improved code:
td = timedelta(days=2, hours=3, minutes=4, seconds=5)
structure = dehydrate_timedelta(td)
print(f"Days: {structure.days}, Hours: {structure.hours}, Minutes: {structure.minutes}, Seconds: {structure.seconds}")