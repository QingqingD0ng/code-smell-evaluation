import argparse
import sys

def parse_arguments(*unparsed_arguments):
    parser = argparse.ArgumentParser(description='Process some arguments.')
    subparsers = parser.add_subparsers(dest='subparser_name')

    # Example of adding a subparser
    subparser_a = subparsers.add_parser('subparser_a')
    subparser_a.add_argument('--option1', help='Option 1 for subparser A')

    subparser_b = subparsers.add_parser('subparser_b')
    subparser_b.add_argument('--option2', help='Option 2 for subparser B')

    # Parse the arguments
    args = parser.parse_args(unparsed_arguments)
    return args

if __name__ == "__main__":
    arguments_dict = parse_arguments(*sys.argv[1:])
    print(arguments_dict)