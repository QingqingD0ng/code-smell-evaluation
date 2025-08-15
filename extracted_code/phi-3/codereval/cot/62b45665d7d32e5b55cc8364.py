def parse_subparser_arguments(unparsed_arguments, subparsers):
    parsed_arguments = {name: parser.parse_args(args) for name, parser in subparsers.items()}
    remaining_arguments = unparsed_arguments
    for arg in remaining_arguments:
        found = False
        for parser in subparsers.values():
            if parser.parse_known_args([arg])[0]:
                parsed_arguments[parser.prog].update(parser.parse_args([arg]))
                found = True
                break
        if not found:
            remaining_arguments.remove(arg)
    return parsed_arguments, remaining_arguments