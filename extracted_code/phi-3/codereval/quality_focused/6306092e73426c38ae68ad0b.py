def validate_min_max_args(self, args, min_val, max_val):
    for arg in args:
        if not (min_val <= arg <= max_val):
            return False
    return True