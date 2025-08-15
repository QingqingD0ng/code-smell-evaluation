class Histogram:
    def __init__(self, size, bin_width=1):
        self.size = size
        self.bin_width = bin_width
        self.histogram = [0] * (size // bin_width)

    def _find_bin(self, coord):
        return int(coord // self.bin_width)

    def fill(self, coord, weight=1):
        if weight < 0:
            raise ValueError("Weight must be non-negative")
        bin_index = self._find_bin(coord)
        if 0 <= bin_index < len(self.histogram):
            self.histogram[bin_index] += weight

    def get_histogram(self):
        return self.histogram

    def __str__(self):
        return str(self.histogram)