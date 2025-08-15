from itertools import zip_longest

class Graph:
    # Assuming Graph has a method get_points that returns an iterable of points
    def get_points(self):
        # This should be implemented to return the graph's points
        pass

def to_csv(graph, separator=",", header=None):
    def convert_point(point):
        return separator.join(str(part) for part in point)

    points = graph.get_points()
    csv_lines = []

    if header is not None:
        csv_lines.append(header)

    for point in points:
        csv_lines.append(convert_point(point))

    return "\n".join(csv_lines)