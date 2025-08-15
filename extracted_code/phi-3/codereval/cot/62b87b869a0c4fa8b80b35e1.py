from typing import Callable, Sequence, Tuple, Union

class Histogram:
    pass

class Graph:
    def __init__(self, points: Sequence[Tuple[float, float]], field_names: Sequence[str]):
        self.points = points
        self.field_names = field_names

def hist_to_graph(hist: Histogram, make_value: Callable = None, get_coordinate: str = "left",
                 field_names: Sequence[str] = ("x", "y"), scale: Union[Callable, None] = None) -> Graph:
    if make_value is None:
        make_value = lambda bin_: bin_

    def get_bin_coordinates(bin_):
        if get_coordinate == "left":
            return bin_.left
        elif get_coordinate == "right":
            return bin_.right
        elif get_coordinate == "middle":
            return (bin_.left + bin_.right) / 2
        else:
            raise ValueError("Invalid get_coordinate value")

    graph_points = []
    for bin_ in hist.bins:  # Assuming hist.bins is an iterable of bin objects
        x = get_bin_coordinates(bin_)
        y = make_value(bin_)
        if scale:
            y = scale(bin_)  # Apply scaling if provided
        graph_points.append((x, y))

    return Graph(graph_points, field_names)