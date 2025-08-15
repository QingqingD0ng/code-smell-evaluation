class ArgumentChecker:
    def __init__(self):
        self.deprecated_args = {
            '--old-arg': 'DEPRECATED: Use --new-arg instead.',
        }

    def validate_arg_deprecation(self, cli_args, answer_file_args):
        all_args = {**cli_args, **answer_file_args}
        for arg, message in self.deprecated_args.items():
            if arg in all_args:
                print(f"Warning: {arg} is deprecated. {message}")
                del all_args[arg]
        return all_args