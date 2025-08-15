def parse_subparser_arguments(unparsed_arguments, subparsers):
    parsed_args = {}
    remaining_args = []

    def add_to_parser(parser, args):
        parsed_namespace = parser.parse_args(args)
        parsed_args[parser.dest] = parsed_namespace
        return parsed_namespace

    for subparser_name, parser in subparsers.items():
        try:
            parsed_namespace = add_to_parser(parser, unparsed_arguments)
            remaining_args = [arg for arg in unparsed_arguments if arg not in parsed_namespace.values()]
        except SystemExit:
            pass

    return parsed_args, remaining_args