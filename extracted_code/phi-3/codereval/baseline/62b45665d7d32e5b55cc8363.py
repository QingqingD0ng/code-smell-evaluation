from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def make_parsers():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', help='sub-command help')

    # Subparser for 'add' command
    parser_add = subparsers.add_parser('add', help='Add something')
    parser_add.add_argument('--item', required=True, help='Item to add')

    # Subparser for 'list' command
    parser_list = subparsers.add_parser('list', help='List items')
    parser_list.add_argument('--all', action='store_true', help='List all items')

    # Subparser for'remove' command
    parser_remove = subparsers.add_parser('remove', help='Remove an item')
    parser_remove.add_argument('--item', required=True, help='Item to remove')

    return parser, subparsers