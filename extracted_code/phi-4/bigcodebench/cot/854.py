from functools import reduce
from itertools import permutations
import math

def task_func(numbers):
    if not isinstance(numbers, list) or not all(isinstance(n, int) for n in numbers):
        raise TypeError("Input must be a list of integers.")
    if any(n < 0 for n in numbers):
        raise ValueError("Numbers must be non-negative.")

    if not numbers:
        return [], []

    perms = list(permutations(numbers))
    factorial_sums = [
        sum(math.factorial(n) for n in perm)
        for perm in perms
    ]

    return factorial_sums, perms