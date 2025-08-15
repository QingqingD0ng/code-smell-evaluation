def validate_requires_args(self, args, required_args_info, optional_args=[], default_values={}):

    missing_args = []

    incorrect_args = {}

    for arg, info in required_args_info.items():

        if arg not in args:

            missing_args.append(arg)

        elif not isinstance(args[arg], info['type']):

            incorrect_args[arg] = (args[arg], info['type'])

    for arg in optional_args:

        if arg in args and not isinstance(args[arg], required_args_info[arg]['type']):

            incorrect_args[arg] = (args[arg], required_args_info[arg]['type'])


    if missing_args:

        raise ValueError(f"Missing required arguments: {', '.join(missing_args)}")

    if incorrect_args:

        raise ValueError(f"Incorrect argument types: {', '.join([f'{k}: expected {v[1]}, got {type(args[k])}' for k, v in incorrect_args.items()])}")


    # Set default values for optional args

    for arg in optional_args:

        if arg not in args:

            args[arg] = default_values.get(arg, None)


    # Additional validation for required args

    for arg in required_args_info:

        if arg in args and args[arg] is not None:

            # Add additional validation logic as needed

            pass


    return args  # Returning the validated and possibly updated args dictionary