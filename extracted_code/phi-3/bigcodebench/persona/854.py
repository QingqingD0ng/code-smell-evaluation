from functools import reduce
from itertools import permutations
import math

def task_func(numbers):
    if not all(isinstance(n, int) for n in numbers):
        raise TypeError("All elements must be integers")
    
    if any(n < 0 for n in numbers):
        raise ValueError("Input numbers must be non-negative")

    def factorial_sum(perm):
        return reduce(lambda x, y: x + y, (math.factorial(n) for n in perm))

    perm_list = list(permutations(numbers))
    fac_list = [factorial_sum(perm) for perm in perm_list]
    return fac_list, perm_list