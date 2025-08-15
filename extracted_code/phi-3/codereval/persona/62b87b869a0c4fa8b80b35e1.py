from typing import Callable, Any, Tuple

def hist_to_graph(hist, make_value=None, get_coordinate="left",
                  field_names=("x", "y"), scale=None):
    if make_value is None:
        make_value = lambda bin_: bin_
    if scale is None:
        scale = lambda bin_: bin_

    x = []
    y = []

    for bin_ in hist:
        x_value = getattr(bin_, field_names[0])
        y_value = make_value(bin_)

        if get_coordinate == "left":
            x.append(x_value)
        elif get_coordinate == "right":
            x.append(x_value + (hist.bin_width,))
        elif get_coordinate == "middle":
            x.append((x_value + hist.bin_width) / 2)
        else:
            raise ValueError("Invalid get_coordinate value")

        y.append(scale(y_value))

    return {'x': x, 'y': y, 'field_names': field_names}