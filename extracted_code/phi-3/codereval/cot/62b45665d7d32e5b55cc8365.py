def parse_arguments(*unparsed_arguments):
    parsed_arguments = {}
    for arg in unparsed_arguments:
        key, value = arg.split('=')
        parsed_arguments[key] = value
    return parsed_arguments