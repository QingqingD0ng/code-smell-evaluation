def validate_requires_args(self, args):
    required_args = self.get_required_args()  # Assuming this method exists
    return all(arg in args for arg in required_args)