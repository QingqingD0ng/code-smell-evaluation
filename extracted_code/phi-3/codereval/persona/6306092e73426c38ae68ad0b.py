class ArgumentValidator:
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def validate_min_max_args(self, *args):
        for arg in args:
            if not (self.min_value <= arg <= self.max_value):
                raise ValueError(f"Argument {arg} is out of the allowed range [{self.min_value}, {self.max_value}]")
        return True

# Usage example:
# validator = ArgumentValidator(min_value=0, max_value=100)
# validator.validate_min_max_args(10, 20, 50)  # Should raise ValueError for the second argument if it's not within the range.