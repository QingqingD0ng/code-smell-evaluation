from functools import reduce
from itertools import permutations
import math

def sum_factorials_of_permutation(permutation):
    """Calculate the sum of the factorials of a permutation."""
    return sum(math.factorial(num) for num in permutation)

def task_func(numbers):
    if not numbers:
        return [], []
    if not all(isinstance(n, int) for n in numbers):
        raise TypeError("All elements must be integers")
    if any(n < 0 for n in numbers):
        raise ValueError("Input numbers cannot be negative")

    perms = list(permutations(numbers))
    sum_factorials = [sum_factorials_of_permutation(perm) for perm in perms]
    return sum_factorials, perms