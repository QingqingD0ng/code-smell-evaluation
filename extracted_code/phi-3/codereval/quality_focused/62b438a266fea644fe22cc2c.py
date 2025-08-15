import argparse

def make_parsers():
    parser = argparse.ArgumentParser(description='Top-level parser with subparsers.')
    subparsers = parser.add_subparsers(dest='command')

    # Example of adding a subparser
    parser_info = subparsers.add_parser('info', help='Display information')
    parser_info.add_argument('--data', type=str, help='Data to display')

    parser_version = subparsers.add_parser('version', help='Show version information')
    parser_version.add_argument('--version', type=str, help='Version to display')

    # Add more subparsers as needed

    # Return the main parser and the subparsers
    return parser, subparsers

# Example usage
if __name__ == '__main__':
    parser, subparsers = make_parsers()
    args = parser.parse_args()
    if args.command == 'info':
        print("Displaying information with data:", args.data)
    elif args.command =='version':
        print("Version information:", args.version)