from lena.context import LenaValueError

class GraphScaler(object):
    def __init__(self, graph):
        self.graph = graph

    def scale(self, other=None):
        if other is None:
            return self.graph.scale

        last_coordinate = self.graph.get_last_coordinate()
        if not last_coordinate:
            raise LenaValueError("Graph has no scale.")

        if not isinstance(other, (int, float)):
            raise LenaValueError("Scale must be a numeric value.")

        self.graph.rescale(last_coordinate, other)
        return self.graph.scale