import warnings
from functools import wraps
from typing import Callable, Any

def deprecated(message: str, version: str = None, replacement: str = None) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def new_func(*args: Any, **kwargs: Any) -> Any:
            full_message = f"Call to {func.__name__} is deprecated."
            if version:
                full_message += f" It will be removed in version {version}."
            if replacement:
                full_message += f" Use {replacement} instead."
            warnings.warn(full_message, category=DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return new_func
    return decorator