def parser_flags(parser):
    flags = []
    for action in parser._actions:
        if not isinstance(action, _ArgumentGroup):
            flags.append(action.dest)
    return ''.join(flags)