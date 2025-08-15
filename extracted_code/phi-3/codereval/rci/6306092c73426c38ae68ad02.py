import json


class ArgumentValidator:

    def __init__(self, deprecated_args_source):

        with open(deprecated_args_source, 'r') as file:

            config = json.load(file)

        self.deprecated_args = set(config['deprecated_args'])


    def validate_arg_deprecation(self, cli_args, answer_file_args):

        all_args = set(cli_args) | set(answer_file_args)

        deprecated_args = self.deprecated_args & all_args

        for arg in deprecated_args:

            print(f"Warning: '{arg}' is deprecated and will be removed in the next version.")

        return deprecated_args


# Example usage:

# Assuming 'deprecated_args.json' contains:

# {

#     "deprecated_args": ["old_arg1", "old_arg2", "old_arg3"]

# }

validator = ArgumentValidator('deprecated_args.json')

cli_args = ['new_arg1', 'old_arg1']

answer_file_args = ['old_arg2', 'new_arg2']

deprecated_args = validator.validate_arg_deprecation(cli_args, answer_file_args)