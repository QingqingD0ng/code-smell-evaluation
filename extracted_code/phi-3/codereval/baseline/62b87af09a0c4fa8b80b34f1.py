class Histogram:
    def __init__(self, size):
        self.size = size
        self.histogram = [0] * size

    def fill(self, coord, weight=1):
        if 0 <= coord < self.size:
            self.histogram[coord] += weight