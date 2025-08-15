def parse_subparser_arguments(unparsed_arguments, subparsers):
    result = {}
    remaining_arguments = []

    for arg in unparsed_arguments:
        parsed = False
        for subparser_name, subparser in subparsers.items():
            namespace, remaining = subparser.parse_known_args([arg], namespace=subparser.parse_args([arg]))
            if namespace:
                result[subparser_name] = namespace
                parsed = True
                break
        if not parsed:
            remaining_arguments.append(arg)

    return result, remaining_arguments