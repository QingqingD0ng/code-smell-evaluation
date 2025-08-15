import numpy as np

def gaussian(x: np.ndarray) -> np.ndarray:
    u: float = 0.2
    sigma: float = 0.1
    return np.exp(-((x - u) ** 2) / (2 * sigma ** 2))

gaussian_values = gaussian(np.array([1, 2, 3]))
print(gaussian_values)