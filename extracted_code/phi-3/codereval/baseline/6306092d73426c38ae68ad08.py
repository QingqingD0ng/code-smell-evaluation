def _get_conditionally_required_args(self, command_name, options_spec, args):
    conditionally_required_args = []
    for option in options_spec:
        if option.get('name') == command_name and option.get('required_when'):
            for arg_name in option['required_when']:
                if arg_name in args:
                    conditionally_required_args.append(arg_name)
    return conditionally_required_args