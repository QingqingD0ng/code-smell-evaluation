import argparse

def parse_arguments(*args_with_flags):
    parser = argparse.ArgumentParser(description='Script arguments parser.')
    for arg in args_with_flags:
        if isinstance(arg, tuple):
            # Tuple format: (flag, help string)
            flag, help_string = arg
            parser.add_argument(flag, help=help_string)
        else:
            # String represents the flag
            parser.add_argument(arg)
    return parser

# Example usage:
if __name__ == "__main__":
    parser = parse_arguments(
        ('--version', 'Show script version'),
        ('--input', 'Input file path'),
        '--output', 'Output file path'
    )
    args = parser.parse_args()
    print(args)