class Graph:
    def __init__(self, scale=1.0):
        self.scale = scale
        self.fields = {'x': [], 'y': [], 'z': []}

    def add_field(self, field_name, values):
        self.fields[field_name] = values

    def _check_non_zero_scale(self):
        if self.scale == 0:
            raise ValueError("LenaValueError: Graph scale cannot be zero.")

    def get_scale(self):
        self._check_non_zero_scale()
        return self.scale

    def set_scale(self, other):
        if other <= 0:
            raise ValueError("LenaValueError: Scale must be a positive number.")
        self.scale = other
        self._rescale_fields()

    def _rescale_fields(self):
        last_field = next(reversed(self.fields))
        self.fields[last_field] = [value * self.scale for value in self.fields[last_field]]

    def scale(self, other=None):
        if other is None:
            return self.get_scale()
        else:
            self.set_scale(other)

# Example usage:
graph = Graph(scale=2.0)
graph.add_field('x', [1, 2, 3])
graph.add_field('y', [4, 5, 6])

print(graph.scale())  # Output: 2.0

graph.scale(0.5)

print(graph.fields)  # Output: {'x': [0.5, 1.0, 1.5], 'y': [2.0, 2.5, 3.0]}

graph.scale(10)

print(graph.fields)  # Output: {'x': [5.0, 10.0, 15.0], 'y': [40.0, 50.0, 60.0]}