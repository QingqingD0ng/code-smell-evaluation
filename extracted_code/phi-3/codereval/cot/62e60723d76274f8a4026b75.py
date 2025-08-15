import math


def round_half_to_even(n):

    floor_val = math.floor(n)

    if n - floor_val < 0.5:

        return floor_val

    else:

        return math.ceil(n - 0.5)