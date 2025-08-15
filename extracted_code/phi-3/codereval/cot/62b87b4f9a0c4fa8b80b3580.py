import numpy as np

def integral(bins, edges):
    bins = np.array(bins)
    edges = np.array(edges)
    bin_widths = np.diff(edges)
    
    if len(bin_widths) == 0 or len(bin_widths) == 1 and bin_widths[0] == 0:
        return 0.0
    
    area = np.sum(bins[:-1] * bin_widths)
    
    return area