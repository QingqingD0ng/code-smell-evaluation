def parse_arguments(*unparsed_arguments):
    args_dict = {}
    for arg in unparsed_arguments:
        if '=' in arg:
            key, value = arg.split('=', 1)
            args_dict[key] = value
        else:
            args_dict[arg] = None
    return args_dict