def validate_requires_args(self, args):
    required_args = ['arg1', 'arg2', 'arg3']  # Example list of required arguments
    missing_args = [arg for arg in required_args if arg not in args]
    if missing_args:
        raise ValueError(f"Missing required arguments: {', '.join(missing_args)}")