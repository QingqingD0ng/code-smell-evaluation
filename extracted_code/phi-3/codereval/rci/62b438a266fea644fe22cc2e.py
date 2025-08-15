import argparse

def parse_arguments(*unparsed_arguments):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subparser_name')

    parser_a = subparsers.add_parser('a')
    parser_a.add_argument('--arg1', type=int)

    parser_b = subparsers.add_parser('b')
    parser_b.add_argument('--arg2', type=str)

    return parser.parse_args(*unparsed_arguments)