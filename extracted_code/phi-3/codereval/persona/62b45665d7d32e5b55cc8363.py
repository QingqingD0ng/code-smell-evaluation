import argparse

def make_parsers():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    parser_sub1 = subparsers.add_parser('sub1')
    parser_sub1.add_argument('arg1', type=int)

    parser_sub2 = subparsers.add_parser('sub2')
    parser_sub2.add_argument('--option', type=str)

    return parser, subparsers