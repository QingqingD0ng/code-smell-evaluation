class Point:
    def __init__(self, coordinates):
        self.coordinates = coordinates

    def __repr__(self):
        return f"Point(coordinates={self.coordinates})"

def create_point_class(name, fields, srid_map):
    class Point:
        def __init__(self, **kwargs):
            for field, value in kwargs.items():
                setattr(self, field, value)
                if field in fields:
                    setattr(self,'srid', srid_map.get(field, None))

        def __repr__(self):
            return f"{name}({', '.join(f'{k}={v}' for k, v in self.__dict__.items() if not k.startswith('_'))})"

    return Point

# Usage example
fields = ['x', 'y']
srid_map = {'x': 'EPSG:4326', 'y': 'EPSG:4326'}
Point = create_point_class('Point', fields, srid_map)

# Creating a point instance
p = Point(x=10, y=20)
print(p)  # Output: Point(x=10, y=20, srid=EPSG:4326)