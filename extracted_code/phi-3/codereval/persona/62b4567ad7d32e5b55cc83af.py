import argparse

def parse_arguments(*arguments):
    parser = argparse.ArgumentParser(description='Process some arguments.')
    for arg in arguments:
        parser.add_argument(arg)
    return parser.parse_args()