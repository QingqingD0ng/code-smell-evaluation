def parse_arguments(*unparsed_arguments):
    arg_dict = {}
    for arg in unparsed_arguments:
        if '=' in arg:
            key, value = arg.split('=', 1)
            arg_dict[key] = value
        else:
            arg_dict[arg] = None
    return arg_dict