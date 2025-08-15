class DehydratedPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def dehydrate_point(value):
    if isinstance(value, dict):
        if 'x' in value and 'y' in value:
            return DehydratedPoint(value['x'], value['y'])
        else:
            raise ValueError("Dictionary must contain 'x' and 'y' keys.")
    else:
        raise TypeError("Input must be a dictionary with 'x' and 'y' keys.")