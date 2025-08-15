import argparse

def parser_flags(parser):
    flags = [action.dest for action in parser._actions if isinstance(action, argparse._StoreAction) or isinstance(action, argparse._StoreConstAction)]
    return ''.join(flags)