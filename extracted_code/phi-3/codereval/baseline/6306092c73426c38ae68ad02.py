class ArgumentValidator:
    def validate_arg_deprecation(self, cli_args, answer_file_args):
        deprecated_args = ['old_arg1', 'old_arg2']  # Example deprecated arguments
        for arg in cli_args:
            if arg in deprecated_args:
                print(f"Warning: '{arg}' is deprecated and will be removed in the next version.")
        for arg in answer_file_args:
            if arg in deprecated_args:
                print(f"Warning: '{arg}' is deprecated and will be removed in the next version.")