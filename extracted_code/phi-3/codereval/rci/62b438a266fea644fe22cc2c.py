import argparse

from collections import defaultdict


def make_parsers():
    parser = argparse.ArgumentParser(description='Main application parser.')
    subparsers_actions = parser.add_subparsers(dest='command', help='Sub-command help')
    subparsers = defaultdict(lambda: subparsers_actions.add_parser('default', help='Default command'))

    def add_subparser(name, help):
        subparsers[name] = subparsers_actions.add_parser(name, help=help)

    add_subparser('run', 'Run the application with optional verbosity.')
    add_subparser('config', 'Manage application configuration.')
    add_subparser('init', 'Initialize application.')
    # Add more subparsers as needed...

    return parser, subparsers.values()