import math

def gaussian(x, u=0.2, sigma=0.1):
    coefficient = 1 / (sigma * math.sqrt(2 * math.pi))
    exponent = math.exp(-0.5 * ((x - u) / sigma) ** 2)
    return coefficient * exponent

# Example usage:
x_value = 0.5
print(gaussian(x_value))