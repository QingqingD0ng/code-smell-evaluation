from functools import reduce
from itertools import permutations
import math

def task_func(numbers):
    if not isinstance(numbers, list) or not all(isinstance(x, int) for x in numbers):
        raise TypeError("Input must be a list of integers.")
    if any(n < 0 for n in numbers):
        raise ValueError("Integers must be non-negative.")
    
    perms = list(permutations(numbers))
    factorial_sums = [reduce(lambda x, y: x + y, (math.factorial(n) for n in perm)) for perm in perms]
    
    return factorial_sums, perms

# Example usage:
# fac, perm = task_func([1, 2, 3])
# print(fac)  # Output: [9, 9, 9, 9, 9, 9]
# print(perm)  # Output: [(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)]