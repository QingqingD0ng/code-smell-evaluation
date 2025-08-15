import numpy as np

def gaussian(x):
    u = 0.2
    sigma = 0.1
    return np.exp(-((x - u) ** 2) / (2 * sigma ** 2))