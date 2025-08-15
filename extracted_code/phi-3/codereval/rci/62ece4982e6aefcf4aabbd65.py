def parser_flags(parser):
    return''.join(f'{action.dest}={action.default}' if action.default is not None else action.dest for action in parser._actions if isinstance(action, argparse._StoreAction))