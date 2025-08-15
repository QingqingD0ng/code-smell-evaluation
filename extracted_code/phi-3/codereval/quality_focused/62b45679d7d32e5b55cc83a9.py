def parser_flags(parser):
    return''.join(f'--{name}' for name in parser._option_string_actions)

# Example usage:
import argparse

# Create the parser
parser = argparse.ArgumentParser()
parser.add_argument('--example', help='An example argument')

# Get the flags
flags = parser_flags(parser)
print(flags)  # Output: --example