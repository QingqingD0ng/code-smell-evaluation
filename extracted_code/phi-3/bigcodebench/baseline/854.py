from functools import reduce
from itertools import permutations
import math

def task_func(numbers):
    if not all(isinstance(n, int) for n in numbers):
        raise TypeError("All elements must be integers")
    if any(n < 0 for n in numbers):
        raise ValueError("Input numbers cannot be negative")
    
    def sum_of_factorials(nums):
        return reduce(lambda x, y: x + math.factorial(y), nums, 0)

    perms = list(permutations(numbers))
    fac = [sum_of_factorials(perm) for perm in perms]
    return fac, perms