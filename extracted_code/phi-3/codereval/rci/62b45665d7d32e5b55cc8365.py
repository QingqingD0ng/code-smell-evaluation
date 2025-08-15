import argparse

def parse_arguments(*unparsed_arguments):
    parser = argparse.ArgumentParser()
    for arg in unparsed_arguments:
        if '=' in arg:
            key, value = arg.split('=', 1)
            parser.add_argument(f'--{key}', type=type(value))
        else:
            parser.add_argument(arg)
    return vars(parser.parse_args(unparsed_arguments))