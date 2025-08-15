class ArgumentValidator:
    def __init__(self, required_args):
        self.required_args = required_args

    def validate_requires_args(self, args):
        missing_args = [arg for arg in self.required_args if arg not in args]
        if missing_args:
            raise ValueError(f"Missing required arguments: {', '.join(missing_args)}")
        return True

# Example usage:
required_args = ['arg1', 'arg2', 'arg3']
validator = ArgumentValidator(required_args)

try:
    validator.validate_requires_args(['arg1', 'arg2'])  # This will raise a ValueError
except ValueError as e:
    print(e)