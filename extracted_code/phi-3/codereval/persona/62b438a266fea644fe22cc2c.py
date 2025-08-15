import argparse

def make_parsers():
    # Create the top-level parser
    parser = argparse.ArgumentParser(description="Main parser")
    
    # Create subparsers for subcommands
    subparsers = parser.add_subparsers(dest='command', help='Subcommands')

    # Example subcommand parser
    sub_parser = subparsers.add_parser('example', help='An example subcommand')
    
    # Add arguments to the subparser
    sub_parser.add_argument('--arg1', type=int, help='Argument 1 for example subcommand')
    sub_parser.add_argument('--arg2', type=str, help='Argument 2 for example subcommand')

    # Return the top-level parser and the subparser as a tuple
    return parser, sub_parser

# Example usage
if __name__ == "__main__":
    main_parser, example_sub_parser = make_parsers()
    args = main_parser.parse_args()
    if args.command == 'example':
        main_parser.print_help()