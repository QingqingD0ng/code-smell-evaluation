import numpy as np


class Histogram:

    def __init__(self, data=None, bins=10):

        self.data = data

        self.bins = bins

        self.histogram = None

        self.scale = None


    def fill(self, data):

        self.data = data

        self.histogram, _ = np.histogram(self.data, bins=self.bins)


    def compute_scale(self):

        if self.scale is None:

            self.scale = np.sum(self.histogram)


    def scale_to(self, other):

        if self.scale == 0 or other == 0:

            raise ValueError("Cannot rescale histogram with zero scale.")

        scale_factor = other / self.scale

        self.histogram *= scale_factor

        self.scale = other


    def scale(self, other=None, recompute=False):

        if other is None:

            if recompute or self.scale is None:

                self.compute_scale()

            return self.scale

        else:

            if not isinstance(other, (int, float)):

                raise TypeError("other must be a float.")

            self.scale_to(other)

            return self.scale


# Example usage:

# hist = Histogram(data=[1, 2, 2, 3, 3, 3])

# print(hist.scale())  # Compute scale

# hist.scale(2.0)       # Rescale to 2.0

# print(hist.scale())  # Should print 2.0