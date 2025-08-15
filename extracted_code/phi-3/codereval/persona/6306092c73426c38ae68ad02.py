def validate_arg_deprecation(self, cli_args, answer_file_args):

    deprecated_args = ["--old-arg", "--legacy-option"]  # Example deprecated arguments


    # Merge dictionaries

    all_args = {**cli_args, **answer_file_args}


    # Check for deprecated arguments

    for arg in deprecated_args:

        if arg in all_args:

            print(f"Warning: '{arg}' is deprecated and will be removed in future versions.")