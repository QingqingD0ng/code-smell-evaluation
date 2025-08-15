class SoftwareQualityExpert:
    def __init__(self, max_length):
        self.max_length = max_length

    def validate_length_args(self, args):
        for arg in args:
            if isinstance(arg, str) and len(arg) > self.max_length:
                return False
        return True

# Example usage:
expert = SoftwareQualityExpert(max_length=10)
args = ["short", "medium", "longerstring"]
print(expert.validate_length_args(args))  # Output: False

args = ["short", "medium", "justshort"]
print(expert.validate_length_args(args))  # Output: True