def get_nested_custom_and_control_args(self, args):
    control_args = {}
    nested_args = {}
    custom_args = {}

    for arg, value in args.items():
        if arg.startswith('--control-'):
            control_args[arg[len('--control-'):]] = value
        elif arg.startswith('--nested-'):
            nested_args[arg[len('--nested-'):]] = value
        elif arg.startswith('--custom-'):
            custom_args[arg[len('--custom-'):]] = value
        else:
            nested_args[arg] = value

    return control_args, nested_args, custom_args