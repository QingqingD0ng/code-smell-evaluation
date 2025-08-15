from functools import lru_cache
from typing import Any, Callable

def lru_cache(maxsize: int = 128, typed: bool = False) -> Callable:
    return lru_cache(maxsize=maxsize, typed=typed)

# Usage example with a custom function
@lru_cache(maxsize=200, typed=True)
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Accessing the cached results
print(fibonacci(10))  # Output: 55
print(fibonacci(10))  # This call will be fetched from the cache