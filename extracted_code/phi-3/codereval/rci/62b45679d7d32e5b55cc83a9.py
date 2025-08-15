def parser_flags(parser):
    return''.join(f'--{action.dest}' for action in parser._actions if action.dest)