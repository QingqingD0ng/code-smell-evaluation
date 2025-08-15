from functools import reduce
from itertools import permutations
import math

def task_func(numbers):
    if not isinstance(numbers, list) or not all(isinstance(n, int) for n in numbers):
        raise TypeError("Input must be a list of integers.")
    if any(n < 0 for n in numbers):
        raise ValueError("Input numbers must not be negative.")

    factorial_sums = [reduce(lambda x, y: x + y, [math.factorial(n) for n in perm]) for perm in permutations(numbers)]
    all_permutations = list(map(list, permutations(numbers)))

    return factorial_sums, all_permutations

# Example usage:
fac, perm = task_func([1, 2, 3])
print(fac)
print(perm)

fac, perm = task_func([0, 4])
print(fac)
print(perm)