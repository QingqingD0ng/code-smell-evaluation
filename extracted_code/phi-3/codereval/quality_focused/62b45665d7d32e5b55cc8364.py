def parse_subparser_arguments(unparsed_arguments, subparsers):
    # Initialize containers for parsed arguments and remaining arguments
    parsed_arguments = {}
    remaining_arguments = []

    # Iterate over the subparsers to parse arguments
    for subparser_name, subparser in subparsers.items():
        # Parse the arguments using the subparser and add to the parsed_arguments
        namespace, unknown_args = subparser.parse_known_args(unparsed_arguments)
        parsed_arguments[subparser_name] = namespace
        # Collect the remaining arguments
        remaining_arguments.extend(unknown_args)
        # Remove the arguments that have been claimed
        unparsed_arguments = remaining_arguments

    # Return the parsed arguments and the remaining arguments that were not claimed
    return parsed_arguments, remaining_arguments