from argparse import ArgumentParser

def parse_subparser_arguments(unparsed_arguments, subparsers):
    parsed_args = {}
    remaining_arguments = list(unparsed_arguments)

    def try_subparser(subparser_name):
        try:
            args, remaining = subparsers[subparser_name].parse_known_args(remaining_arguments)
            if args:
                parsed_args[subparser_name] = args
            return remaining
        except SystemExit:
            return remaining_arguments

    while remaining_arguments:
        for subparser_name, subparser in subparsers.items():
            remaining_arguments = try_subparser(subparser_name)
            if not remaining_arguments:
                break

    return parsed_args, remaining_arguments