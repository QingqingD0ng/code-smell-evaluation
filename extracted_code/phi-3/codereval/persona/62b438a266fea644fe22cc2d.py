import argparse

def parse_subparser_arguments(unparsed_arguments, subparsers):
    # Create a new ArgumentParser to handle shared arguments
    shared_parser = argparse.ArgumentParser(add_help=False)
    for action in shared_parser.add_argument_group("shared_arguments").actions:
        for subparser in subparsers.values():
            subparser.add_argument(*action.option_strings, **action.kwargs)

    # Parse the shared arguments
    shared_namespace, shared_remainders = shared_parser.parse_known_args(unparsed_arguments)

    # Initialize the result dictionaries
    subparser_results = {}
    remaining_arguments = unparsed_arguments[:]

    # Parse each subparser's arguments
    for subparser_name, subparser in subparsers.items():
        try:
            subparser_namespace, subparser_remainders = subparser.parse_known_args(remaining_arguments)
            subparser_results[subparser_name] = subparser_namespace
            remaining_arguments = subparser_remainders
        except SystemExit:
            # Skip subparsers that exit the program
            pass

    return subparser_results, remaining_arguments