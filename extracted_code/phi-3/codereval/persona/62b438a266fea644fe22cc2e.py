import argparse
import sys

def parse_arguments(*unparsed_arguments):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='parser_name', help='sub-command help')

    # Example subparser
    parser_a = subparsers.add_parser('a', help='do something A')
    parser_a.add_argument('--foo', help='foo help')

    # Add more subparsers as needed

    # Parse the command line arguments
    args = parser.parse_args(unparsed_arguments)

    # Create a dictionary to store subparser names and their corresponding Namespace instances
    parsed_args = {}

    # If a subparser was invoked
    if args.parser_name:
        # Check if the subparser is already in the dictionary
        if args.parser_name in parsed_args:
            raise argparse.ArgumentError(None, f"Duplicate subparser {args.parser_name}")
        # Add the subparser name and its Namespace instance to the dictionary
        parsed_args[args.parser_name] = args

    return parsed_args

if __name__ == "__main__":
    # Command-line arguments are passed to this example script
    args = parse_arguments(*sys.argv[1:])
    print(args)