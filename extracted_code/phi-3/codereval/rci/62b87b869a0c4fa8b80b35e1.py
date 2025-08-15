import numpy as np

class Histogram:
    def __init__(self, bins, edges):
        self.bins = bins
        self.edges = edges

def hist_to_graph(hist, make_value=None, get_coordinate="left", field_names=("x", "y"), scale=None):
    if not isinstance(hist, Histogram):
        raise ValueError("hist must be an instance of Histogram")
    
    if make_value is None:
        make_value = lambda bin_: (bin_, bin_)
    
    if get_coordinate not in ["left", "right", "middle"]:
        raise ValueError("get_coordinate must be 'left', 'right', or'middle'")
    
    if len(field_names)!= 2:
        raise ValueError("field_names must contain exactly two field names")
    
    x_field, y_field = field_names
    
    if scale is True:
        scale_factor = (hist.edges[1] - hist.edges[0]) / len(hist.bins)
    elif scale is not None:
        scale_factor = scale
    else:
        scale_factor = 1
    
    graph = []
    
    for i, bin_ in enumerate(hist.bins):
        if isinstance(bin_, tuple):
            value = make_value(bin_)
        else:
            value = make_value(bin_)
        
        if get_coordinate == "left":
            x_value = hist.edges[i]
        elif get_coordinate == "middle":
            x_value = (hist.edges[i] + hist.edges[i+1]) / 2
        else:  # get_coordinate == "right"
            x_value = hist.edges[i+1]
        
        graph.append((x_value,) + value)
    
    dtype = [(x_field, float), (y_field, float)]
    return np.array(graph, dtype=dtype)

# Example usage:
# hist = Histogram(bins=[1, 2