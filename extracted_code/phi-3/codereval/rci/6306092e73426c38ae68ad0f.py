def parse_args(self, args):
    control_args = {}
    nested_args = {}

    for arg in args:
        if arg.startswith('--control-'):
            control_args[arg.split('--control-')[1]] = args[arg]
        elif arg.startswith('--nested-'):
            nested_args[arg.split('--nested-')[1]] = args[arg]
        else:
            if arg not in nested_args and arg not in control_args:
                nested_args[arg] = args[arg]

    return control_args, nested_args