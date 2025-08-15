from argparse import ArgumentParser

def make_parsers():
    parser = ArgumentParser(description='This is a main parser.')
    subparsers = parser.add_subparsers(dest='subparser_name', help='Subcommands')

    # Example: Adding a subparser for a 'run' command
    run_parser = subparsers.add_parser('run', help='Run the application')
    run_parser.add_argument('--config', type=str, help='Path to configuration file')

    # Example: Adding a subparser for a 'list' command
    list_parser = subparsers.add_parser('list', help='List items')
    list_parser.add_argument('--all', action='store_true', help='List all items')

    # You can add more subparsers here following the same pattern

    return parser, subparsers

# Example usage
if __name__ == "__main__":
    parser, subparsers = make_parsers()
    args = parser.parse_args()
    if args.subparser_name == 'run':
        # Handle 'run' subcommand
        pass
    elif args.subparser_name == 'list':
        # Handle 'list' subcommand
        pass