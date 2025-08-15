class InvalidPointError(Exception):
    """Custom exception for invalid point representations."""

class MissingKeyError(Exception):
    """Custom exception for missing required keys in the point representation."""

class InvalidValueTypeError(Exception):
    """Custom exception for invalid value types."""

class DehydratedPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def dehydrate_point(point_representation):
    if not isinstance(point_representation, dict):
        raise TypeError("Input must be a dictionary.")
    required_keys = {'x', 'y'}
    if not required_keys.issubset(point_representation):
        missing_keys = required_keys - set(point_representation.keys())
        raise MissingKeyError(f"Dictionary must contain {', '.join(map(str, missing_keys))} keys.")
    try:
        x = float(point_representation['x'])
        y = float(point_representation['y'])
    except ValueError:
        raise InvalidValueTypeError("'x' and 'y' values must be convertible to numbers.")
    return DehydratedPoint(x, y)