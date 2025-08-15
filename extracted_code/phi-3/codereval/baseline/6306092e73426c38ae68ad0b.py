def validate_min_max_args(self, args):
    if not (isinstance(args, list) and len(args) == 2 and all(isinstance(arg, int) for arg in args)):
        raise ValueError("args must be a list with two integers")

    min_val, max_val = args
    if min_val > max_val:
        raise ValueError("min_val must be less than or equal to max_val")

    for arg in args:
        if not (min_val <= arg <= max_val):
            return False
    return True