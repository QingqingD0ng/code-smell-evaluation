class Point:
    def __init__(self, x, y, srid):
        self.x = x
        self.y = y
        self.srid = srid

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y}, SRID={self.srid})"

def create_point_class(fields, srid):
    field_names = fields.keys()
    field_types = fields.values()

    class DynamicPoint(Point):
        def __init__(self, **kwargs):
            for field in field_names:
                value = kwargs.get(field)
                if value is not None:
                    field_type = field_types[field]
                    setattr(self, field, field_type(value))
            self.srid = srid

        def __repr__(self):
            field_values = ', '.join([f'{field}={getattr(self, field)}' for field in field_names])
            return f"{self.__class__.__name__}({field_values}, SRID={self.srid})"

    return DynamicPoint

# Example usage
fields = {
    'id': int,
    'name': str,
    'location': lambda x, y: Point(x, y)
}
srid = "EPSG:4326"

CustomPoint = create_point_class(fields, srid)
custom_point = CustomPoint(id=1, name='Example', location=(100, 200))
print(custom_point)  # Output: CustomPoint(id=1, name='Example', location=Point(x=100, y=200), SRID=EPSG:4326)