class Time:
    def __init__(self, ticks):
        self.ticks = ticks

    def dehydrate(self):
        return self.ticks

class Structure:
    def __init__(self, time):
        self.time = time

    def dehydrate(self):
        return self.time.dehydrate()