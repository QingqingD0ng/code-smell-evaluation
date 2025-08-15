class Point:
    def __init__(self, name, fields, srid_map):
        self.name = name
        self.fields = fields
        self.srid_map = srid_map
        self.values = {field: None for field in fields}

    def set_value(self, field, value):
        if field in self.fields:
            self.values[field] = value
        else:
            raise ValueError(f"Field {field} does not exist.")

    def get_value(self, field):
        if field in self.fields:
            return self.values[field]
        else:
            raise ValueError(f"Field {field} does not exist.")

    def __repr__(self):
        return f"Point(name={self.name}, fields={self.fields}, srid_map={self.srid_map})"

# Example usage:
point = Point(name="Central Park", fields=["x", "y"], srid_map={"x": "EPSG:4326", "y": "EPSG:4326"})
point.set_value("x", 40.785091)
point.set_value("y", -73.968285)
print(point)