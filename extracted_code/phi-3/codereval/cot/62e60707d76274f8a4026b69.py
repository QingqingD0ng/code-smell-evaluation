class Point:
    def __init__(self, coord):
        self.coord = coord

    def __str__(self):
        return f"Point({self.coord})"

    def __eq__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return self.coord == other.coord

    def __repr__(self):
        return self.__str__()

def point_type(name, fields, srid_map):
    return type(name, (Point,), {})

# Example usage:
CustomPoint = point_type("CustomPoint", {}, srid_map={})
print(CustomPoint(coord=(10, 20)))