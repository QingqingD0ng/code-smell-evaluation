from functools import reduce
from itertools import permutations
import math

def task_func(numbers):
    if not all(isinstance(n, int) for n in numbers):
        raise TypeError("Input must be a list of integers")
    if any(n < 0 for n in numbers):
        raise ValueError("Input numbers cannot be negative")

    perms = list(permutations(numbers))
    factorial_sums = [reduce(lambda x, y: x + math.factorial(y), p, 0) for p in perms]

    return factorial_sums, perms