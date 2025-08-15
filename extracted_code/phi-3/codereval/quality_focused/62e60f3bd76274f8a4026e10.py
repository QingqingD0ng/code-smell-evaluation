from datetime import timedelta

class Structure:
    def __init__(self, days, seconds):
        self.days = days
        self.seconds = seconds

def dehydrate_timedelta(value):
    days, seconds = divmod(value.total_seconds(), 86400)
    return Structure(int(days), int(seconds))