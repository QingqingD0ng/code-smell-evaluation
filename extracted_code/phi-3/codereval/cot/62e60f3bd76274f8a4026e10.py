class TimedeltaStructure:
    def __init__(self, days, seconds):
        self.days = days
        self.seconds = seconds

    def __repr__(self):
        return f"TimedeltaStructure(days={self.days}, seconds={self.seconds})"

def dehydrate_timedelta(value):
    days = value.days
    seconds = value.seconds
    return TimedeltaStructure(days, seconds)