class DehydratedPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def dehydrate_point(value):
    if not isinstance(value, (list, tuple)) or len(value)!= 2:
        raise ValueError("Input must be a list or tuple with exactly two elements.")
    return DehydratedPoint(*value)