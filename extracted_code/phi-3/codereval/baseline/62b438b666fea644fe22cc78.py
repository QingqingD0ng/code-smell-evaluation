import argparse

def parse_arguments(*arguments):
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    for arg in arguments:
        parser.add_argument(arg)
    return parser.parse_args()