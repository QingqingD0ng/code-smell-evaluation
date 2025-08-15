class Time:
    def __init__(self, ticks):
        self.ticks = ticks
        self.seconds = self.ticks / 1000
        self.minutes = self.seconds / 60
        self.hours = self.minutes / 60
        self.days = self.hours / 24

    def __str__(self):
        return f"{self.days:.2f} days, {self.hours:.2f} hours, {self.minutes:.2f} minutes, {self.seconds:.2f} seconds"

class Structure:
    def __init__(self, value):
        self.value = value
        self.time = Time(value)

    def dehydrate_time(self):
        return self.time