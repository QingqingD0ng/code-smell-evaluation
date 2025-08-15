control_args = {}
nested_args = {}

def get_nested_custom_and_control_args(self, args):
    for arg in args:
        if arg.startswith('--control-'):
            control_args[arg.split('--control-')[1]] = args[arg]
        elif arg.startswith('--nested-'):
            nested_args[arg.split('--nested-')[1]] = args[arg]
        else:
            custom_arg = arg
            if custom_arg not in nested_args and custom_arg not in control_args:
                nested_args[custom_arg] = args[arg]