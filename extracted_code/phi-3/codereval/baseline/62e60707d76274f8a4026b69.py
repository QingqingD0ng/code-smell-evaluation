class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"

def point_type(name, fields, srid_map):
    field_names = fields.keys()
    field_types = fields.values()
    srid = srid_map.get(name, "SRID 4326")

    class DynamicPoint(Point):
        def __init__(self, **kwargs):
            for field in field_names:
                setattr(self, field, kwargs.get(field))

        def __repr__(self):
            return f"{name}({', '.join([f'{field}={getattr(self, field)}' for field in field_names])})"

    return DynamicPoint

# Example usage:
fields = {
    'id': int,
    'name': str,
    'location': Point
}
srid_map = {
    'custom_point': 'SRID 4326'
}

CustomPoint = point_type('CustomPoint', fields, srid_map)
point = CustomPoint(id=1, name='Example', location=Point(100, 200))
print(point)  # Output: CustomPoint(id=1, name='Example', location=Point(x=100, y=200))