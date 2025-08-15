class ArgumentValidator:
    def validate_arg_deprecation(self, cli_args, answer_file_args):
        deprecated_args = self._get_deprecated_args_from_args(cli_args, answer_file_args)
        if deprecated_args:
            for arg, old_value in deprecated_args.items():
                print(f"Warning: The argument '{arg}' with value '{old_value}' is deprecated.")
    
    def _get_deprecated_args_from_args(self, cli_args, answer_file_args):
        deprecated_args = {}
        deprecated_keys = self._get_deprecated_keys()
        for key in deprecated_keys:
            old_value = cli_args.pop(key, None)
            if old_value:
                deprecated_args[key] = old_value
            answer_file_value = answer_file_args.pop(key, None)
            if answer_file_value:
                deprecated_args[key] = answer_file_value
        return deprecated_args
    
    def _get_deprecated_keys(self):
        return ['--old-arg1', '--old-arg2']  # Replace with actual deprecated argument names

# Usage:
# validator = ArgumentValidator()
# validator.validate_arg_deprecation(cli_args, answer_file_args)