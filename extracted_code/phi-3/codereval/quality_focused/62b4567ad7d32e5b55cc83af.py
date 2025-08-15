import argparse

def parse_arguments(*arguments):
    parser = argparse.ArgumentParser(description="Parse command-line arguments.")
    for arg, value in arguments:
        if isinstance(value, list):
            parser.add_argument(arg, metavar='value', nargs='*', type=type(value[0]), help=str(value[0]))
        else:
            parser.add_argument(arg, type=type(value), help=str(value))
    return parser