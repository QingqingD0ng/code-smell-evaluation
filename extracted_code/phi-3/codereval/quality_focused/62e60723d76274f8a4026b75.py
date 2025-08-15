import math


def round_half_to_even(n):

    return round(n + math.copysign(0.5, n))