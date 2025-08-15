from functools import wraps
from typing import Callable, Optional, Dict, Any

def unit_of_work(metadata: Optional[Dict[str, Any]] = None, timeout: Optional[int] = None) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        wrapper.metadata = metadata if metadata is not None else {}
        wrapper.timeout = timeout
        return wrapper
    return decorator

# Example usage:
@unit_of_work(metadata={'description': 'Example task'}, timeout=10)
def example_task():
    pass