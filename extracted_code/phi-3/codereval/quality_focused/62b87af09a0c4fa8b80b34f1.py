class Histogram:
    def __init__(self, size):
        self.histogram = [0] * size

    def fill(self, coord, weight=1):
        if 0 <= coord < len(self.histogram):
            self.histogram[coord] += weight