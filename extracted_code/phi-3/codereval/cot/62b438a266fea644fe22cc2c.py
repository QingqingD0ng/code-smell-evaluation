from argparse import ArgumentParser

def make_parsers():
    def create_subparser(name, help):
        subparsers = parser.add_subparsers(dest='subcommand')
        subparser = subparsers.add_parser(name, help=help)
        return subparser

    parser = ArgumentParser()
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    subparsers = {
      'subparser1': create_subparser('subparser1', 'Subparser 1 help message'),
      'subparser2': create_subparser('subparser2', 'Subparser 2 help message'),
    }

    return parser, subparsers