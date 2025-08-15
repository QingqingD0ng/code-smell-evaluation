from datetime import timedelta

class Structure:
    def __init__(self, days, seconds):
        self.days = days
        self.seconds = seconds

def dehydrate_timedelta(value):
    total_seconds = int(value.total_seconds())
    days, remainder = divmod(total_seconds, 86400)
    seconds = remainder % 60
    minutes, remainder = divmod(remainder, 60)
    hours, minutes = divmod(minutes, 60)
    return Structure(days, hours, minutes, seconds)

# Example usage:
td = timedelta(days=2, hours=3, minutes=4, seconds=5)
structure = dehydrate_timedelta(td)
print(f"Days: {structure.days}, Hours: {structure.hours}, Minutes: {structure.minutes}, Seconds: {structure.seconds}")