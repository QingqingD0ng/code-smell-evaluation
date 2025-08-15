import argparse

def parse_arguments(*arguments):
    parser = argparse.ArgumentParser()
    for arg in arguments:
        if isinstance(arg, tuple) and len(arg) == 2:
            parser.add_argument(*arg)
        else:
            parser.add_argument(arg)
    return parser