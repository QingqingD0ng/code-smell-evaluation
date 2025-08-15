from functools import reduce
from itertools import permutations
import math

def task_func(numbers):
    if not isinstance(numbers, list) or not all(isinstance(n, int) for n in numbers):
        raise TypeError("Input must be a list of integers.")
    if any(n < 0 for n in numbers):
        raise ValueError("Integers must be non-negative.")

    perm_list = list(permutations(numbers))
    factorial_sums = [
        reduce(lambda acc, x: acc + math.factorial(x), perm, 0)
        for perm in perm_list
    ]

    return factorial_sums, perm_list