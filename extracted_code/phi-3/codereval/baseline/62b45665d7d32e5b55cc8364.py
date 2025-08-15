from argparse import ArgumentParser

def parse_subparser_arguments(unparsed_arguments, subparsers):
    remaining_arguments = unparsed_arguments.copy()
    parsed_args = {}

    while remaining_arguments:
        for subparser_name, subparser in subparsers.items():
            try:
                args, remaining = subparser.parse_known_args(remaining_arguments)
                if args:
                    parsed_args[subparser_name] = args
                    remaining_arguments = remaining
                    break
            except SystemExit:
                # Ignore SystemExit, subparsers will handle it
                pass

    return parsed_args, remaining_arguments