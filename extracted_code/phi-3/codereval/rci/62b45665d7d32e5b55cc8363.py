import argparse

def create_parser(subparsers, help_defaults=True):
    formatter_class = argparse.ArgumentDefaultsHelpFormatter if help_defaults else argparse.HelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter_class)
    subparsers = parser.add_subparsers(dest='command', help='sub-command help')
    return parser, subparsers

def setup_add_parser(subparsers):
    add_parser = subparsers.add_parser('add', help='Add something')
    add_parser.add_argument('--item', required=True, help='Item to add')

def setup_list_parser(subparsers):
    list_parser = subparsers.add_parser('list', help='List items')
    list_parser.add_argument('--all', action='store_true', help='List all items')

def setup_remove_parser(subparsers):
    remove_parser = subparsers.add_parser('remove', help='Remove an item')
    remove_parser.add_argument('--item', required=True, help='Item to remove')

def make_parsers():
    parser, subparsers = create_parser(subparsers=subparsers)
    setup_add_parser(subparsers)
    setup_list_parser(subparsers)
    setup_remove_parser(subparsers)
    return parser, subparsers