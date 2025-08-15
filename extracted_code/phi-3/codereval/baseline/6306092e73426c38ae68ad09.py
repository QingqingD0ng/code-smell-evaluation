class ArgumentValidator:
    def __init__(self, max_length):
        self.max_length = max_length

    def validate_length_args(self, args):
        for arg in args:
            if len(arg) > self.max_length:
                return False
        return True

# Example usage:
# validator = ArgumentValidator(10)
# args = ['short','medium','very_long_string_that_exceeds_the_limit']
# is_valid = validator.validate_length_args(args)
# print(is_valid)  # This will print False