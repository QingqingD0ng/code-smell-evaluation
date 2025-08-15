import numpy as np
from typing import Tuple

def create_zero_filled_array(shape: Tuple[int,...], dtype: np.dtype = np.dtype('float32')) -> np.ndarray:
    try:
        return np.zeros(shape, dtype)
    except TypeError as e:
        raise TypeError(f"Invalid input for'shape': {shape}. Expected a tuple of ints.") from e
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}") from e