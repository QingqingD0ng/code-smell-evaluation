def parser_flags(parser):
    return''.join(f'{action.dest}={action.default}' for action in parser._actions if action.dest is not None)