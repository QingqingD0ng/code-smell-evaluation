class ArgumentValidator:
    def __init__(self, max_length):
        self.max_length = max_length

    def validate_length_args(self, *args):
        for arg in args:
            if isinstance(arg, str) and len(arg) > self.max_length:
                return False
            elif not isinstance(arg, str) and len(arg) > self.max_length:
                return False
        return True

# Example usage:
validator = ArgumentValidator(max_length=10)
print(validator.validate_length_args("valid_str", "another_valid_str", 10))  # True
print(validator.validate_length_args("too_long_str", "short_str"))  # False