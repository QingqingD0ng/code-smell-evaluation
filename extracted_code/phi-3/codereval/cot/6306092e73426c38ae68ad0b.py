class ArgumentValidator:
    def __init__(self, min_value=None, max_value=None):
        self.min_value = min_value
        self.max_value = max_value

    def validate_min_max_args(self, args):
        if not (self.min_value <= args <= self.max_value if self.min_value and self.max_value is not None else False):
            raise ValueError(f"Argument {args} is out of the allowed range ({self.min_value}, {self.max_value}).")

# Usage example:
validator = ArgumentValidator(min_value=10, max_value=20)
validator.validate_min_max_args(15)  # This will pass
validator.validate_min_max_args(25)  # This will raise a ValueError