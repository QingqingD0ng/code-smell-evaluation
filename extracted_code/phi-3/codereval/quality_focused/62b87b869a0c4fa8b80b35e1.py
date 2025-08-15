from typing import Callable, Any, Tuple

class Histogram:
    # Assuming Histogram class has some properties and methods
    pass

class Graph:
    # Assuming Graph class has some properties and methods
    pass

def hist_to_graph(hist: Histogram, make_value: Callable[[Any], Tuple[Any, Any]],
                  get_coordinate: str = "left", field_names: Tuple[str,...] = ("x", "y"),
                  scale: bool = None) -> Graph:
    graph = Graph()
    coordinates = {field: [] for field in field_names}

    for bin in hist.bins:
        if not bin.is_numeric():
            raise ValueError("Bins must be numeric.")

        value = bin.content if make_value is None else make_value(bin)

        if get_coordinate == "left":
            coord = bin.left
        elif get_coordinate == "right":
            coord = bin.right
        elif get_coordinate == "middle":
            coord = (bin.left + bin.right) / 2
        else:
            raise ValueError("Invalid get_coordinate value.")

        for i, field in enumerate(field_names):
            coordinates[field].append(value[i] if i < len(value) else None)

    # Assuming Graph class has a method to set coordinates
    graph.set_coordinates(coordinates)

    if scale is not None:
        graph.set_scale(scale)

    return graph