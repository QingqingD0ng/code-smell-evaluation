class Time:
    def __init__(self, ticks):
        self.ticks = ticks

class Structure:
    def __init__(self, time):
        self.time = time

def dehydrate_time(value):
    time = Time(value)
    structure = Structure(time)
    return structure