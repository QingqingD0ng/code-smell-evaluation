import math

def gaussian(x):
    u = 0.2
    sigma = 0.1
    return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - u) / sigma) ** 2)