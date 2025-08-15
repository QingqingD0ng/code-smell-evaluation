def validate_min_max_args(self, args):
    if not (isinstance(args, list) and len(args) == 2 and all(isinstance(arg, int) for arg in args)):
        raise ValueError("args must be a list containing exactly two integers")

    min_val, max_val = args
    if min_val is None or max_val is None or min_val > max_val:
        raise ValueError("args must contain exactly two integers in ascending order")

    for arg in args:
        if not isinstance(arg, int):
            raise ValueError("Each argument must be an integer")
        if arg < min_val or arg > max_val:
            return False
    return True