class ArgumentValidator:
    def __init__(self, max_length):
        self.max_length = max_length

    def validate_length(self, args):
        if all(len(arg) <= self.max_length for arg in args):
            return True
        return False

# Example usage:
# validator = ArgumentValidator(max_length=10)
# result = validator.validate_length(['short','medium', 'longer'])