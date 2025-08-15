class Graph:
    def __init__(self):
        self.scale = 1.0  # Initialize scale to 1.0 (no scaling)

    def _rescale_error(self, error_coords):
        raise LenaValueError(f"Cannot rescale graph with zero or unknown scale: {error_coords}")

    def _get_last_coordinate(self, coordinates):
        # Assuming coordinates is a list of tuples or lists
        return coordinates[-1] if coordinates else None

    def _set_scale(self, scale_value):
        if scale_value <= 0:
            self._rescale_error(self._get_last_coordinate(self.coordinates))
        self.scale = scale_value

    def scale(self, other=None):
        if other is None:
            return self.scale
        else:
            self._set_scale(other)
            self.coordinates = [
                tuple(scale_value * self.scale for scale_value in coord)
                for coord in self.coordinates
            ]

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, value):
        self._coordinates = value

# Assuming LenaValueError is defined somewhere in the code
class LenaValueError(ValueError):
    pass