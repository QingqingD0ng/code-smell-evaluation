def get_nested_custom_and_control_args(self, args):
    control_args = {}
    nested_args = {}
    custom_args = {}
    
    for arg in args:
        if 'control' in arg:
            control_args[arg] = args[arg]
        elif 'nested' in arg:
            nested_args[arg] = args[arg]
        elif 'custom' in arg:
            custom_args[arg] = args[arg]
        else:
            raise ValueError(f"Unknown argument type for {arg}")
    
    return control_args, nested_args, custom_args