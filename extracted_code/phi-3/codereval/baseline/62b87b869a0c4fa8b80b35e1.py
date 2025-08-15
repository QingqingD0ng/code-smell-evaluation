import numpy as np

def hist_to_graph(hist, make_value=None, get_coordinate="left", field_names=("x", "y"), scale=None):
    if make_value is None:
        make_value = lambda bin_: (bin_, bin_)
    
    if get_coordinate not in ["left", "right", "middle"]:
        raise ValueError("get_coordinate must be 'left', 'right', or'middle'")
    
    if len(field_names)!= 2:
        raise ValueError("field_names must contain exactly two field names")
    
    x_field, y_field = field_names
    graph = []
    
    if scale is True:
        scale_factor = (hist.bins[1] - hist.bins[0]) / len(hist.bins)
    else:
        scale_factor = 1
    
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
    
    return np.array(graph)