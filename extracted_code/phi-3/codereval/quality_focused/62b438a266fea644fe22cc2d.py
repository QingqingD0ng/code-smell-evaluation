def parse_subparser_arguments(unparsed_arguments, subparsers):
    parsed_args = {}
    remaining_args = []
    subparser_names = []

    while unparsed_arguments:
        arg = unparsed_arguments.pop(0)
        if arg.startswith('--'):
            subparser_name, *argument_parts = arg[2:].split('=', 1)
            subparser = subparsers.get(subparser_name)
            if subparser:
                if argument_parts:
                    parsed_args[subparser_name] = subparser.parse_args(argument_parts)
                else:
                    parsed_args[subparser_name] = subparser.parse_args()
                subparser_names.append(subparser_name)
            else:
                remaining_args.append(arg)
        else:
            remaining_args.append(arg)

    return parsed_args, remaining_args