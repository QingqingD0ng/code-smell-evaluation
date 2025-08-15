def get_nested_custom_and_control_args(self, args):
    control_args = {}
    nested_args = {}

    for arg in args:
        if arg.startswith('--control-'):
            control_key = arg.split('--control-')[1]
            control_args[control_key] = arg
        elif arg.startswith('--nested-'):
            nested_key = arg.split('--nested-')[1]
            nested_args[nested_key] = arg
        else:
            nested_args[arg] = arg

    return control_args, nested_args