import numpy as np
from typing import Any

def force_string(obj: Any) -> str:
    if isinstance(obj, (bytes, np.bytes_)):
        try:
            return obj.decode('utf-8')
        except UnicodeDecodeError as e:
            raise ValueError(f"Decoding error: {e}") from e
    return str(obj)